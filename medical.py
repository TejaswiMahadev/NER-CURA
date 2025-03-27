import os
import json
import re
import tempfile
import base64
from io import BytesIO

import dash
from dash import dcc, html, Input, Output, State, ctx, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# Load API keys
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# ------------------- PDF Processing -------------------

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks"""
    if not text_chunks:
        raise ValueError("No text chunks provided for embeddings.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error creating FAISS vector store: {e}")

def extract_medical_entities(text):
    """
    Extracts detailed medical entities using Gemini with improved categorization
    and context awareness
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Extract and categorize medical entities from the following text with high precision.
    Use context clues to properly identify entities when possible.
    
    Categories:
    - Medications: prescription drugs, OTC medications, supplements
    - Medical_Conditions: diseases, disorders, symptoms, diagnoses
    - Procedures: surgeries, therapeutic procedures, diagnostic tests
    - Lab_Results: test values, metrics, measurements, biomarkers
    - Vital_Signs: blood pressure, heart rate, temperature, etc.
    - Anatomical_Structures: organs, body parts, systems
    - Medical_Devices: implants, assistive devices, monitoring equipment
    - Healthcare_Providers: specialists, physicians, therapists

    For each entity, extract:
    1. The exact term as it appears in text
    2. The normalized/standardized term when applicable
    3. Any associated values or measurements
    4. Temporal information when available (date, duration, frequency)
    
    Return ONLY valid JSON. Do NOT include explanations, code blocks, or extra text.

    Example Output:
    {{
      "medications": [
        {{
          "term": "Lisinopril",
          "normalized": "lisinopril",
          "dosage": "10mg",
          "frequency": "daily",
          "start_date": "2023-01-15"
        }}
      ],
      "medical_conditions": [
        {{
          "term": "T2DM",
          "normalized": "Type 2 Diabetes Mellitus",
          "diagnosis_date": "2020-03-10"
        }}
      ],
      "procedures": [...],
      "lab_results": [...],
      "vital_signs": [...],
      "anatomical_structures": [...],
      "medical_devices": [...],
      "healthcare_providers": [...]
    }}

    Text: "{text}"
    """

    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    raw_text = re.sub(r"```json\s*([\s\S]+?)\s*```", r"\1", raw_text)

    try:
        entities = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Error: Response is not valid JSON")
        print("Raw Output:", raw_text)
        return {}

    return entities

def identify_entity_relationships(entities):
    """
    Identifies relationships between medical entities
    such as medications treating conditions, procedures for diagnoses, etc.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Convert entities to string for prompt
    entity_json = json.dumps(entities, indent=2)
    
    prompt = f"""
    Analyze these medical entities and identify potential relationships between them:
    
    {entity_json}
    
    Identify relationships such as:
    1. Which medications treat which conditions
    2. Which procedures are related to which diagnoses
    3. Which lab results are associated with which conditions
    4. Which providers are treating which conditions
    
    Return ONLY valid JSON. Format the output as:
    
    {{
      "medication_condition_relations": [
        {{
          "medication": "Metformin",
          "treats_condition": "Type 2 Diabetes Mellitus",
          "confidence": "high"
        }}
      ],
      "procedure_condition_relations": [...],
      "lab_condition_relations": [...],
      "provider_condition_relations": [...]
    }}
    """
    
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    raw_text = re.sub(r"```json\s*([\s\S]+?)\s*```", r"\1", raw_text)

    try:
        relationships = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Error: Response is not valid JSON")
        print("Raw Output:", raw_text)
        return {}

    return relationships

def highlight_entities(text, entities):
    """
    Highlights medical entities in text with improved styling
    and tooltips for additional context
    """
    # Create a copy of text to avoid modification issues
    highlighted_text = text
    
    # Define styling for each category with tooltips
    color_map = {
        "medications": {"color": "#0277bd", "icon": "💊"},  # Blue
        "medical_conditions": {"color": "#c62828", "icon": "🩺"},  # Red
        "procedures": {"color": "#2e7d32", "icon": "⚕️"},  # Green
        "lab_results": {"color": "#6a1b9a", "icon": "🧪"},  # Purple
        "vital_signs": {"color": "#ef6c00", "icon": "📊"},  # Orange
        "anatomical_structures": {"color": "#3e2723", "icon": "🫀"},  # Brown
        "medical_devices": {"color": "#546e7a", "icon": "🔧"},  # Gray-blue
        "healthcare_providers": {"color": "#1a237e", "icon": "👩‍⚕️"}  # Indigo
    }
    
    # Process each category and its entities
    for category, style in color_map.items():
        if category not in entities:
            continue
            
        for entity_data in entities[category]:
            # Handle both string entities and dictionary entities
            if isinstance(entity_data, str):
                entity = entity_data
                tooltip = f"{style['icon']} {category.replace('_', ' ').title()}"
            else:
                entity = entity_data.get("term", "")
                
                # Create detailed tooltip with available information
                tooltip_parts = [f"{style['icon']} {category.replace('_', ' ').title()}"]
                
                if "normalized" in entity_data and entity_data["normalized"] != entity:
                    tooltip_parts.append(f"Normalized: {entity_data['normalized']}")
                
                for key, value in entity_data.items():
                    if key not in ["term", "normalized"] and value:
                        tooltip_parts.append(f"{key.replace('_', ' ').title()}: {value}")
                
                tooltip = " | ".join(tooltip_parts)
            
            # Escape entity for regex and create highlighting with tooltip
            if entity:
                escaped_entity = re.escape(entity)
                replacement = f'<span style="background-color: {style["color"]}20; color: {style["color"]}; font-weight: bold; padding: 1px 4px; border-radius: 3px; border: 1px solid {style["color"]}80;" title="{tooltip}">{entity}</span>'
                
                # Use word boundary for whole-word matching when possible
                highlighted_text = re.sub(rf"\b{escaped_entity}\b", replacement, highlighted_text)
    
    return highlighted_text

def get_confidence_color(confidence):
    """Returns a color based on confidence level"""
    if confidence.lower() == "high":
        return "#1b5e20"
    elif confidence.lower() == "medium":
        return "#f57f17"
    else:
        return "#c62828"

def generate_summary(text):
    """
    Generates a summary of the medical record text using Gemini
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Generate a comprehensive medical summary of the following medical record.
    Focus on key diagnoses, medications, treatments, and important clinical findings.
    Organize the information clearly and concisely.
    
    Text: "{text}"
    """

    response = model.generate_content(prompt)
    return response.text.strip()

def user_input(question, raw_text):
    """
    Processes user questions about the medical record and returns relevant answers
    """
    try:
        # Process text if no vector store exists
        if not os.path.exists("faiss_index"):
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            
        # Load vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings)
        
        # Get relevant documents
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate answer using Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        Context from medical records:
        {context}
        
        Question: {question}
        
        Please answer the question based on the provided context from the medical records.
        Be clear, accurate, and provide specific information from the records when available.
        If the answer is not in the context, say "I don't see information about that in the medical records."
        """
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Extract any medical entities in the response
        entities = extract_medical_entities(answer)
        
        return answer, entities
    except Exception as e:
        return f"Error processing your question: {str(e)}", {}

def extract_temporal_data(entities):
    """
    Extracts temporal data from entities for timeline visualization
    """
    timeline_data = []
    
    # Process medications with dates
    for entity in entities.get("medications", []):
        if isinstance(entity, dict):
            # Get the medication name
            medication = entity.get("term", "")
            
            # Check for date fields
            start_date = entity.get("start_date", "")
            end_date = entity.get("end_date", "")
            
            if medication and start_date:
                timeline_item = {
                    "type": "medication",
                    "name": medication,
                    "start_date": start_date,
                    "end_date": end_date if end_date else None,
                    "color": "#0277bd"
                }
                timeline_data.append(timeline_item)
    
    # Process conditions with dates
    for entity in entities.get("medical_conditions", []):
        if isinstance(entity, dict):
            # Get the condition name
            condition = entity.get("term", "")
            
            # Check for date fields
            diagnosis_date = entity.get("diagnosis_date", "")
            resolution_date = entity.get("resolution_date", "")
            
            if condition and diagnosis_date:
                timeline_item = {
                    "type": "condition",
                    "name": condition,
                    "start_date": diagnosis_date,
                    "end_date": resolution_date if resolution_date else None,
                    "color": "#c62828"
                }
                timeline_data.append(timeline_item)
    
    # Process procedures with dates
    for entity in entities.get("procedures", []):
        if isinstance(entity, dict):
            # Get the procedure name
            procedure = entity.get("term", "")
            
            # Check for date fields
            procedure_date = entity.get("date", "")
            
            if procedure and procedure_date:
                timeline_item = {
                    "type": "procedure",
                    "name": procedure,
                    "start_date": procedure_date,
                    "end_date": None,  # Procedures are typically single events
                    "color": "#2e7d32"
                }
                timeline_data.append(timeline_item)
    
    return timeline_data

def create_entity_network_graph(entities, relationships):
    """Creates a NetworkX graph for Plotly visualization"""
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Node lists by type for later
    nodes_by_type = {
        "medications": [],
        "medical_conditions": [],
        "procedures": []
    }
    
    # Add nodes for entities
    added_nodes = set()
    
    # Add nodes for medications
    for entity in entities.get("medications", []):
        if isinstance(entity, str):
            node_id = entity
        else:
            node_id = entity.get("term", "")
        
        if node_id and node_id not in added_nodes:
            G.add_node(node_id, color="#0277bd", group="medications")
            added_nodes.add(node_id)
            nodes_by_type["medications"].append(node_id)
    
    # Add nodes for conditions
    for entity in entities.get("medical_conditions", []):
        if isinstance(entity, str):
            node_id = entity
        else:
            node_id = entity.get("term", "")
        
        if node_id and node_id not in added_nodes:
            G.add_node(node_id, color="#c62828", group="medical_conditions")
            added_nodes.add(node_id)
            nodes_by_type["medical_conditions"].append(node_id)
    
    # Add nodes for procedures
    for entity in entities.get("procedures", []):
        if isinstance(entity, str):
            node_id = entity
        else:
            node_id = entity.get("term", "")
        
        if node_id and node_id not in added_nodes:
            G.add_node(node_id, color="#2e7d32", group="procedures")
            added_nodes.add(node_id)
            nodes_by_type["procedures"].append(node_id)
    
    # Add edges for relationships
    edges = []
    
    if relationships:
        # Add medication-condition relationships
        for relation in relationships.get("medication_condition_relations", []):
            med = relation.get("medication", "")
            cond = relation.get("treats_condition", "")
            if med and cond and med in added_nodes and cond in added_nodes:
                G.add_edge(med, cond, weight=1)
                edges.append((med, cond))
        
        # Add procedure-condition relationships
        for relation in relationships.get("procedure_condition_relations", []):
            proc = relation.get("procedure", "")
            cond = relation.get("for_condition", "")
            if proc and cond and proc in added_nodes and cond in added_nodes:
                G.add_edge(proc, cond, weight=1)
                edges.append((proc, cond))
    
    # Get node positions
    pos = nx.spring_layout(G)
    
    # Create node traces for Plotly
    node_traces = {}
    
    # Colors for different node types
    colors = {
        "medications": "#0277bd",
        "medical_conditions": "#c62828",
        "procedures": "#2e7d32"
    }
    
    # Create a trace for each node type
    for node_type, nodes in nodes_by_type.items():
        if not nodes:
            continue
            
        x_vals = []
        y_vals = []
        text_vals = []
        
        for node in nodes:
            if node in pos:
                position = pos[node]
                x_vals.append(position[0])
                y_vals.append(position[1])
                text_vals.append(node)
        
        node_traces[node_type] = go.Scatter(
            x=x_vals,
            y=y_vals,
            text=text_vals,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                size=15,
                color=colors[node_type],
                line=dict(width=2, color="white")
            ),
            name=node_type.replace("_", " ").title()
        )
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="Relationships"
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace] + list(node_traces.values()),
        layout=go.Layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

# ------------------- Main NER Functions -------------------
def ner_processing(text):
    """
    Main function for NER processing that returns all extracted data
    """
    # Extract entities
    entities = extract_medical_entities(text)
    
    # Identify relationships between entities
    relationships = identify_entity_relationships(entities)
    
    # Extract temporal data for timeline
    timeline_data = extract_temporal_data(entities)
    
    return entities, relationships, timeline_data

# ------------------- Dash UI Components -------------------

# Function to create color legends
def create_color_legend():
    colors = {
        "medications": {"color": "#0277bd", "icon": "💊", "title": "Medications"},
        "medical_conditions": {"color": "#c62828", "icon": "🩺", "title": "Medical Conditions"},
        "procedures": {"color": "#2e7d32", "icon": "⚕️", "title": "Procedures"},
        "lab_results": {"color": "#6a1b9a", "icon": "🧪", "title": "Lab Results"},
        "vital_signs": {"color": "#ef6c00", "icon": "📊", "title": "Vital Signs"},
        "anatomical_structures": {"color": "#3e2723", "icon": "🫀", "title": "Anatomical Structures"},
        "medical_devices": {"color": "#546e7a", "icon": "🔧", "title": "Medical Devices"},
        "healthcare_providers": {"color": "#1a237e", "icon": "👩‍⚕️", "title": "Healthcare Providers"}
    }
    
    legend_items = []
    
    for category, info in colors.items():
        legend_items.append(
            html.Div([
                html.Div(style={
                    "width": "20px",
                    "height": "20px",
                    "backgroundColor": info["color"],
                    "marginRight": "10px",
                    "borderRadius": "3px",
                    "display": "inline-block"
                }),
                html.Div(f"{info['icon']} {info['title']}", style={
                    "fontWeight": 500,
                    "display": "inline-block"
                })
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})
        )
    
    return html.Div(legend_items, style={
        "padding": "15px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "5px",
        "border": "1px solid #dee2e6"
    })

# Entity display component
def create_entity_card(entity_data, category, color, icon):
    if isinstance(entity_data, str):
        return html.Div([
            html.Span(entity_data, style={
                "color": color,
                "fontWeight": 500
            })
        ], style={
            "backgroundColor": f"{color}10",
            "padding": "5px 8px",
            "marginBottom": "5px",
            "borderRadius": "3px"
        })
    else:
        # Display rich entity data
        entity_term = entity_data.get("term", "")
        
        # Additional data items
        data_items = []
        
        # Add normalized term if available and different
        if "normalized" in entity_data and entity_data["normalized"] != entity_term:
            data_items.append(html.Div([
                html.Span("Normalized: ", style={"color": "#555", "fontWeight": 500}),
                html.Span(entity_data["normalized"])
            ], style={"fontSize": "0.9em", "marginTop": "3px"}))
        
        # Add other fields
        for key, value in entity_data.items():
            if key not in ["term", "normalized"] and value:
                key_formatted = key.replace('_', ' ').title()
                
                # Format values differently based on field type
                if "date" in key.lower() and isinstance(value, str):
                    value_formatted = [html.Span("📅 "), html.Span(value)]
                elif "dosage" in key.lower() and isinstance(value, str):
                    value_formatted = [html.Span("💊 "), html.Span(value)]
                elif "frequency" in key.lower() and isinstance(value, str):
                    value_formatted = [html.Span("🕒 "), html.Span(value)]
                else:
                    value_formatted = value
                    
                data_items.append(html.Div([
                    html.Span(f"{key_formatted}: ", style={"color": "#555", "fontWeight": 500}),
                    html.Span(value_formatted)
                ], style={"fontSize": "0.9em", "marginTop": "2px"}))
        
        return html.Div([
            html.Div(entity_term, style={
                "color": color,
                "fontWeight": 600,
                "fontSize": "1.05em"
            }),
            html.Div(data_items)
        ], style={
            "backgroundColor": f"{color}10",
            "padding": "8px",
            "marginBottom": "8px",
            "borderRadius": "3px",
            "borderLeft": f"3px solid {color}"
        })

# Create entity section
def create_entity_section(entities, category, color, icon, title):
    if category not in entities or not entities[category]:
        return html.Div()
    
    entities_in_category = entities[category]
    count = len(entities_in_category)
    
    return html.Div([
        html.H4([
            html.Span(f"{icon} {title} ({count})"),
        ], style={"color": color, "marginTop": 0}),
        html.Div([
            create_entity_card(entity_data, category, color, icon)
            for entity_data in entities_in_category
        ], style={"maxHeight": "300px", "overflowY": "auto"})
    ], style={
        "padding": "10px",
        "border": f"1px solid {color}",
        "borderRadius": "5px",
        "marginBottom": "15px"
    })

# Create relationship display
def create_relationship_display(relationships):
    if not relationships:
        return html.Div("No relationships found")
    
    sections = []
    
    # Medication-condition relationships
    if "medication_condition_relations" in relationships and relationships["medication_condition_relations"]:
        med_cond_items = []
        
        for relation in relationships["medication_condition_relations"]:
            confidence = relation.get("confidence", "medium")
            confidence_color = get_confidence_color(confidence)
            
            med_cond_items.append(html.Div([
                html.Div(relation.get("medication", ""), style={
                    "backgroundColor": "#0277bd20",
                    "color": "#0277bd",
                    "fontWeight": "bold",
                    "padding": "3px 8px",
                    "borderRadius": "3px",
                    "border": "1px solid #0277bd80"
                }),
                html.Div("➡️", style={"margin": "0 10px"}),
                html.Div(relation.get("treats_condition", ""), style={
                    "backgroundColor": "#c6282820",
                    "color": "#c62828",
                    "fontWeight": "bold",
                    "padding": "3px 8px",
                    "borderRadius": "3px",
                    "border": "1px solid #c6282880"
                }),
                html.Div(f"({confidence} confidence)", style={
                    "marginLeft": "10px",
                    "color": confidence_color,
                    "fontSize": "0.8em"
                })
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}))
        
        sections.append(html.Div([
            html.H4("💊➡️🩺 Medications treating Conditions"),
            html.Div(med_cond_items)
        ]))
    
    # Add other relationship sections based on the relationship types
    relation_configs = [
        {
            "key": "procedure_condition_relations",
            "title": "⚕️➡️🩺 Procedures for Conditions",
            "entity1_key": "procedure", 
            "entity2_key": "for_condition",
            "color1": "#2e7d32", 
            "color2": "#c62828"
        },
        {
            "key": "lab_condition_relations",
            "title": "🧪➡️🩺 Lab Results associated with Conditions",
            "entity1_key": "lab_result", 
            "entity2_key": "associated_condition",
            "color1": "#6a1b9a", 
            "color2": "#c62828"
        },
        {
            "key": "provider_condition_relations",
            "title": "👩‍⚕️➡️🩺 Providers treating Conditions",
            "entity1_key": "provider", 
            "entity2_key": "treats_condition",
            "color1": "#1a237e", 
            "color2": "#c62828"
        }
    ]
    
    for config in relation_configs:
        if config["key"] in relationships and relationships[config["key"]]:
            items = []
            
            for relation in relationships[config["key"]]:
                confidence = relation.get("confidence", "medium")
                confidence_color = get_confidence_color(confidence)
                
                items.append(html.Div([
                    html.Div(relation.get(config["entity1_key"], ""), style={
                        "backgroundColor": f"{config['color1']}20",
                        "color": config["color1"],
                        "fontWeight": "bold",
                        "padding": "3px 8px",
                        "borderRadius": "3px",
                        "border": f"1px solid {config['color1']}80"
                    }),
                    html.Div("➡️", style={"margin": "0 10px"}),
                    html.Div(relation.get(config["entity2_key"], ""), style={
                        "backgroundColor": f"{config['color2']}20",
                        "color": config["color2"],
                        "fontWeight": "bold",
                        "padding": "3px 8px",
                        "borderRadius": "3px",
                        "border": f"1px solid {config['color2']}80"
                    }),
                    html.Div(f"({confidence} confidence)", style={
                        "marginLeft": "10px",
                        "color": confidence_color,
                        "fontSize": "0.8em"
                    })
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}))
            
            sections.append(html.Div([
                html.H4(config["title"]),
                html.Div(items)
            ]))
    
    return html.Div(sections)

# ------------------- Initialize Dash App -------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📄 Medical Record AI Assistant", className="mt-3"),
            html.P("Upload and analyze your medical records. The AI will assist you in understanding your health data."),
        ], width=12)
    ])
    dbc.Row([
        # Left sidebar for color legend and stats
        dbc.Col([
            html.H3("Entity Color Legend", className="mt-3"),
            create_color_legend(),
            
            html.Div(id="document-statistics", className="mt-4"),
            
        ], width=3),
        
        # Main content area
        dbc.Col([
            dbc.Tabs([
                # Tab 1: Text Summary
                dbc.Tab([
                    html.H3("Text Summary", className="mt-3"),
                    dcc.Upload(
                        id='upload-pdf-summary',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select PDF Files')
                        ])
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'marginBottom': '10px',
            'cursor': 'pointer',
        }),
        multiple=False
    ),
    html.Div(id='pdf-text-display', style={'margin': '20px 0'}),
    html.Div(id='pdf-summary', style={'margin': '20px 0'}),
], label="Summary"),

# Tab 2: Entity Recognition
dbc.Tab([
    html.H3("Entity Recognition", className="mt-3"),
    dcc.Upload(
        id='upload-pdf-ner',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select PDF Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'marginBottom': '10px',
            'cursor': 'pointer',
        }),
        multiple=False
    ),
    html.Div(id='entity-extraction-results'),
], label="Entities"),

# Tab 3: Relationships
dbc.Tab([
    html.H3("Entity Relationships", className="mt-3"),
    html.Div(id='relationship-display'),
    html.Div(id='network-graph', className="mt-4"),
], label="Relationships"),

# Tab 4: Timeline
dbc.Tab([
    html.H3("Medical Timeline", className="mt-3"),
    html.Div(id='timeline-display'),
], label="Timeline"),

# Tab 5: Q&A
dbc.Tab([
    html.H3("Ask Questions", className="mt-3"),
    dbc.Input(id="question-input", placeholder="Ask a question about the medical record...", type="text"),
    dbc.Button("Ask", id="ask-button", color="primary", className="mt-2"),
    html.Div(id="answer-output", className="mt-3"),
], label="Q&A"),
], className="mt-3")
], width=9)
]),

# Store components for data sharing between callbacks
dcc.Store(id="pdf-text-storage"),
dcc.Store(id="entities-storage"),
dcc.Store(id="relationships-storage"),
dcc.Store(id="timeline-storage"),

], fluid=True)

# ------------------- Callbacks -------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📄 Medical Record AI Assistant", className="mt-3"),
            html.P("Upload and analyze your medical records. The AI will assist you in understanding your health data."),
        ], width=12)
    ]),
    
    dbc.Row([
        # Left sidebar for color legend and stats
        dbc.Col([
            html.H3("Entity Color Legend", className="mt-3"),
            create_color_legend(),
            
            html.Div(id="document-statistics", className="mt-4"),
            
        ], width=3),
        
        # Main content area
        dbc.Col([
            dbc.Tabs([
                # Tab 1: Text Summary
                dbc.Tab([
                    html.H3("Text Summary", className="mt-3"),
                    dcc.Upload(
                        id='upload-pdf-summary',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select PDF Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'marginBottom': '10px',
                            'cursor': 'pointer',
                        },
                        multiple=False
                    ),
                    html.Div(id='pdf-text-display', style={'margin': '20px 0'}),
                    html.Div(id='pdf-summary', style={'margin': '20px 0'}),
                ], label="Summary"),

                # Tab 2: Entity Recognition
                dbc.Tab([
                    html.H3("Entity Recognition", className="mt-3"),
                    dcc.Upload(
                        id='upload-pdf-ner',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select PDF Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'marginBottom': '10px',
                            'cursor': 'pointer',
                        },
                        multiple=False
                    ),
                    html.Div(id='entity-extraction-results'),
                ], label="Entities"),

                # Tab 3: Relationships
                dbc.Tab([
                    html.H3("Entity Relationships", className="mt-3"),
                    html.Div(id='relationship-display'),
                    html.Div(id='network-graph', className="mt-4"),
                ], label="Relationships"),

                # Tab 4: Timeline
                dbc.Tab([
                    html.H3("Medical Timeline", className="mt-3"),
                    html.Div(id='timeline-display'),
                ], label="Timeline"),

                # Tab 5: Q&A
                dbc.Tab([
                    html.H3("Ask Questions", className="mt-3"),
                    dbc.Input(id="question-input", placeholder="Ask a question about the medical record...", type="text"),
                    dbc.Button("Ask", id="ask-button", color="primary", className="mt-2"),
                    html.Div(id="answer-output", className="mt-3"),
                ], label="Q&A"),
            ], className="mt-3")
        ], width=9)
    ]),

    # Store components for data sharing between callbacks
    dcc.Store(id="pdf-text-storage"),
    dcc.Store(id="entities-storage"),
    dcc.Store(id="relationships-storage"),
    dcc.Store(id="timeline-storage"),

], fluid=True)

# ------------------- Callbacks -------------------

# Callback for PDF upload and summary
@app.callback(
    [Output("pdf-text-storage", "data"),
     Output("pdf-text-display", "children"),
     Output("pdf-summary", "children")],
    Input("upload-pdf-summary", "contents"),
    prevent_initial_call=True
)
def update_summary(contents):
    if contents is None:
        raise PreventUpdate
        
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Process the PDF
    text = get_pdf_text([decoded])
    
    # Generate summary
    summary = generate_summary(text)
    
    # Display PDF text with highlighting
    display_text = html.Div([
        html.H4("Original Text:"),
        html.Div(text, style={
            "maxHeight": "300px", 
            "overflowY": "auto",
            "border": "1px solid #ddd",
            "padding": "10px",
            "borderRadius": "5px",
            "backgroundColor": "#f9f9f9"
        })
    ])
    
    # Display summary
    summary_display = html.Div([
        html.H4("AI Summary:"),
        html.Div(summary, style={
            "padding": "15px",
            "border": "1px solid #4caf50",
            "borderRadius": "5px",
            "backgroundColor": "#f1f8e9"
        })
    ])
    
    return text, display_text, summary_display

# Callback for entity extraction
@app.callback(
    [Output("entities-storage", "data"),
     Output("relationships-storage", "data"),
     Output("timeline-storage", "data"),
     Output("entity-extraction-results", "children")],
    Input("upload-pdf-ner", "contents"),
    prevent_initial_call=True
)
def update_entity_extraction(contents):
    if contents is None:
        raise PreventUpdate
        
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Process the PDF
    text = get_pdf_text([decoded])
    
    # Extract entities and relationships
    entities, relationships, timeline_data = ner_processing(text)
    
    # Create highlighted text display
    highlighted_text = highlight_entities(text, entities)
    
    # Create entity display
    entity_cards = html.Div([
        html.H4("Extracted Medical Entities:"),
        html.Div(highlighted_text, style={
            "maxHeight": "200px", 
            "overflowY": "auto",
            "border": "1px solid #ddd",
            "padding": "10px",
            "borderRadius": "5px",
            "backgroundColor": "#f9f9f9",
            "marginBottom": "20px"
        }),
        html.Div([
            create_entity_section(entities, "medications", "#0277bd", "💊", "Medications"),
            create_entity_section(entities, "medical_conditions", "#c62828", "🩺", "Medical Conditions"),
            create_entity_section(entities, "procedures", "#2e7d32", "⚕️", "Procedures"),
            create_entity_section(entities, "lab_results", "#6a1b9a", "🧪", "Lab Results"),
            create_entity_section(entities, "vital_signs", "#ef6c00", "📊", "Vital Signs"),
            create_entity_section(entities, "anatomical_structures", "#3e2723", "🫀", "Anatomical Structures"),
            create_entity_section(entities, "medical_devices", "#546e7a", "🔧", "Medical Devices"),
            create_entity_section(entities, "healthcare_providers", "#1a237e", "👩‍⚕️", "Healthcare Providers")
        ])
    ])
    
    return entities, relationships, timeline_data, entity_cards

# Callback for relationship display
@app.callback(
    [Output("relationship-display", "children"),
     Output("network-graph", "children")],
    [Input("relationships-storage", "data"),
     Input("entities-storage", "data")],
    prevent_initial_call=True
)
def update_relationships(relationships, entities):
    if not relationships or not entities:
        raise PreventUpdate
    
    # Create relationship display
    relationship_section = html.Div([
        html.H4("Entity Relationships:"),
        create_relationship_display(relationships)
    ])
    
    # Create network graph
    network_fig = create_entity_network_graph(entities, relationships)
    
    network_section = html.Div([
        html.H4("Network Visualization:"),
        dcc.Graph(figure=network_fig, style={"height": "600px"})
    ])
    
    return relationship_section, network_section

# Callback for timeline display
@app.callback(
    Output("timeline-display", "children"),
    Input("timeline-storage", "data"),
    prevent_initial_call=True
)
def update_timeline(timeline_data):
    if not timeline_data:
        return html.Div("No timeline data available. Upload a document with dates.")
    
    # Sort timeline data by date
    timeline_data.sort(key=lambda x: x["start_date"])
    
    # Create timeline visualization using Plotly
    fig = go.Figure()
    
    # Add timeline items
    for item in timeline_data:
        start_date = item["start_date"]
        end_date = item["end_date"] if item["end_date"] else start_date
        
        # Create different shapes based on item type
        if item["type"] == "medication":
            marker_symbol = "pill"
        elif item["type"] == "condition":
            marker_symbol = "cross"
        elif item["type"] == "procedure":
            marker_symbol = "star"
        else:
            marker_symbol = "circle"
        
        # Add the item to the timeline
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[item["name"], item["name"]],
            mode="lines+markers",
            name=item["name"],
            line=dict(color=item["color"], width=2),
            marker=dict(color=item["color"], size=10, symbol=marker_symbol),
            text=[f"{item['type']}: {item['name']} (start)", f"{item['type']}: {item['name']} (end)"],
            hoverinfo="text"
        ))
    
    # Configure layout
    fig.update_layout(
        title="Medical Timeline",
        xaxis_title="Date",
        yaxis_title="Event",
        hovermode="closest",
        height=600
    )
    
    return dcc.Graph(figure=fig)

# Callback for Q&A
@app.callback(
    Output("answer-output", "children"),
    [Input("ask-button", "n_clicks")],
    [State("question-input", "value"),
     State("pdf-text-storage", "data")],
    prevent_initial_call=True
)
def answer_question(n_clicks, question, text):
    if not n_clicks or not question or not text:
        raise PreventUpdate
    
    # Process the question and generate answer
    answer, entities = user_input(question, text)
    
    # Highlight entities in the answer
    highlighted_answer = highlight_entities(answer, entities) if entities else answer
    
    return html.Div([
        html.H4("Answer:"),
        html.Div(dcc.Markdown(highlighted_answer, dangerously_allow_html=True), style={
            "padding": "15px",
            "border": "1px solid #2196f3",
            "borderRadius": "5px",
            "backgroundColor": "#e3f2fd"
        })
    ])

# Callback for document statistics
@app.callback(
    Output("document-statistics", "children"),
    [Input("entities-storage", "data")],
    prevent_initial_call=True
)
def update_statistics(entities):
    if not entities:
        raise PreventUpdate
    
    # Calculate entity counts
    entity_counts = {}
    total_entities = 0
    
    for category, items in entities.items():
        count = len(items)
        entity_counts[category] = count
        total_entities += count
    
    # Create statistics display
    stats = html.Div([
        html.H3("Document Statistics"),
        html.P(f"Total Entities: {total_entities}"),
        html.Hr(),
        html.Div([
            html.Div([
                html.Span(f"{category.replace('_', ' ').title()}: "),
                html.Span(f"{count}", style={"fontWeight": "bold"})
            ]) for category, count in entity_counts.items() if count > 0
        ])
    ], style={
        "padding": "15px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "5px",
        "border": "1px solid #dee2e6"
    })
    
    return stats

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)