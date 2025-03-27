import os
import json
import re
  
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx
import tempfile
    
import streamlit as st
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
        pdf_reader = PdfReader(pdf)
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

# def highlight_entities(text, entities):
#     """
#     Highlights medical entities in text with improved styling
#     and tooltips for additional context
#     """
#     # Create a copy of text to avoid modification issues
#     highlighted_text = text
    
#     # Define styling for each category with tooltips
#     color_map = {
#         "medications": {"color": "#0277bd", "icon": "💊"},  # Blue
#         "medical_conditions": {"color": "#c62828", "icon": "🩺"},  # Red
#         "procedures": {"color": "#2e7d32", "icon": "⚕️"},  # Green
#         "lab_results": {"color": "#6a1b9a", "icon": "🧪"},  # Purple
#         "vital_signs": {"color": "#ef6c00", "icon": "📊"},  # Orange
#         "anatomical_structures": {"color": "#3e2723", "icon": "🫀"},  # Brown
#         "medical_devices": {"color": "#546e7a", "icon": "🔧"},  # Gray-blue
#         "healthcare_providers": {"color": "#1a237e", "icon": "👩‍⚕️"}  # Indigo
#     }
    
#     # Process each category and its entities
#     for category, style in color_map.items():
#         if category not in entities:
#             continue
            
#         for entity_data in entities[category]:
#             # Handle both string entities and dictionary entities
#             if isinstance(entity_data, str):
#                 entity = entity_data
#                 tooltip = f"{style['icon']} {category.replace('_', ' ').title()}"
#             else:
#                 entity = entity_data.get("term", "")
                
#                 # Create detailed tooltip with available information
#                 tooltip_parts = [f"{style['icon']} {category.replace('_', ' ').title()}"]
                
#                 if "normalized" in entity_data and entity_data["normalized"] != entity:
#                     tooltip_parts.append(f"Normalized: {entity_data['normalized']}")
                
#                 for key, value in entity_data.items():
#                     if key not in ["term", "normalized"] and value:
#                         tooltip_parts.append(f"{key.replace('_', ' ').title()}: {value}")
                
#                 tooltip = " | ".join(tooltip_parts)
            
#             # Escape entity for regex and create highlighting with tooltip
#             if entity:
#                 escaped_entity = re.escape(entity)
#                 replacement = f'<span style="background-color: {style["color"]}20; color: {style["color"]}; font-weight: bold; padding: 1px 4px; border-radius: 3px; border: 1px solid {style["color"]}80;" title="{tooltip}">{entity}</span>'
                
#                 # Use word boundary for whole-word matching when possible
#                 highlighted_text = re.sub(rf"\b{escaped_entity}\b", replacement, highlighted_text)
    
#     return highlighted_text

def highlight_entities(text, entities):
    """
    Highlights medical entities in text with improved context handling
    """
    # Create a copy to work with
    processed_text = text
    
    # Process each category
    for category in entities:
        for entity_data in entities[category]:
            if isinstance(entity_data, dict):
                entity = entity_data.get("term", "")
                normalized = entity_data.get("normalized", entity)
            else:
                entity = str(entity_data)
                normalized = entity

            if entity:
                # Create tooltip with context
                tooltip_parts = [f"{category.replace('_', ' ').title()}"]
                if normalized != entity:
                    tooltip_parts.append(f"Normalized: {normalized}")
                
                # Add additional context from entity data
                if isinstance(entity_data, dict):
                    for key, value in entity_data.items():
                        if key not in ["term", "normalized"] and value:
                            tooltip_parts.append(f"{key.title()}: {value}")
                
                tooltip = " | ".join(tooltip_parts)
                
                # Create highlight span
                highlight = f'<span style="background-color: rgba(2, 119, 189, 0.1); border-bottom: 2px dotted #0277bd;" title="{tooltip}">{entity}</span>'
                
                # Replace whole words only
                processed_text = re.sub(
                    rf"\b{re.escape(entity)}\b", 
                    highlight, 
                    processed_text
                )
    
    return processed_text

def get_confidence_color(confidence):
    """Returns a color based on confidence level"""
    if confidence.lower() == "high":
        return "#1b5e20"
    elif confidence.lower() == "medium":
        return "#f57f17"
    else:
        return "#c62828"

def display_entity_lists(entities, relationships=None):
    """
    Displays lists of entities in a structured format with improved visualization
    and relationship information
    """
    # Define styling for each category
    categories = {
        "medications": {"color": "#0277bd", "icon": "💊", "title": "Medications"},
        "medical_conditions": {"color": "#c62828", "icon": "🩺", "title": "Medical Conditions"},
        "procedures": {"color": "#2e7d32", "icon": "⚕️", "title": "Procedures"},
        "lab_results": {"color": "#6a1b9a", "icon": "🧪", "title": "Lab Results"},
        "vital_signs": {"color": "#ef6c00", "icon": "📊", "title": "Vital Signs"},
        "anatomical_structures": {"color": "#3e2723", "icon": "🫀", "title": "Anatomical Structures"},
        "medical_devices": {"color": "#546e7a", "icon": "🔧", "title": "Medical Devices"},
        "healthcare_providers": {"color": "#1a237e", "icon": "👩‍⚕️", "title": "Healthcare Providers"}
    }
    
    # Create tabs for better organization of entities
    entity_tabs = st.tabs(["Main Entities", "Clinical Data", "Healthcare Context", "Relationships"])
    
    with entity_tabs[0]:
        # First tab: Medications and Medical Conditions
        col1, col2 = st.columns(2)
        
        # Medications
        with col1:
            display_category("medications", entities, categories)
        
        # Medical Conditions
        with col2:
            display_category("medical_conditions", entities, categories)
    
    with entity_tabs[1]:
        # Second tab: Procedures and Lab Results, Vital Signs
        col1, col2 = st.columns(2)
        
        # Procedures
        with col1:
            display_category("procedures", entities, categories)
            display_category("vital_signs", entities, categories)
        
        # Lab Results
        with col2:
            display_category("lab_results", entities, categories)
    
    with entity_tabs[2]:
        # Third tab: Anatomical Structures, Medical Devices, Healthcare Providers
        col1, col2 = st.columns(2)
        
        # Anatomical Structures and Medical Devices
        with col1:
            display_category("anatomical_structures", entities, categories)
            display_category("healthcare_providers", entities, categories)
        
        # Healthcare Providers
        with col2:
            display_category("medical_devices", entities, categories)
    
    with entity_tabs[3]:
        # Fourth tab: Entity Relationships
        if relationships:
            st.subheader("Entity Relationships")
            
            # Display medication-condition relationships
            if "medication_condition_relations" in relationships and relationships["medication_condition_relations"]:
                st.markdown("#### 💊➡️🩺 Medications treating Conditions")
                for relation in relationships["medication_condition_relations"]:
                    confidence_color = get_confidence_color(relation.get("confidence", "medium"))
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="background-color: #0277bd20; color: #0277bd; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #0277bd80;">{relation.get("medication", "")}</div>
                        <div style="margin: 0 10px;">➡️</div>
                        <div style="background-color: #c6282820; color: #c62828; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #c6282880;">{relation.get("treats_condition", "")}</div>
                        <div style="margin-left: 10px; color: {confidence_color}; font-size: 0.8em;">({relation.get("confidence", "medium")} confidence)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display procedure-condition relationships
            if "procedure_condition_relations" in relationships and relationships["procedure_condition_relations"]:
                st.markdown("#### ⚕️➡️🩺 Procedures for Conditions")
                for relation in relationships["procedure_condition_relations"]:
                    confidence_color = get_confidence_color(relation.get("confidence", "medium"))
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="background-color: #2e7d3220; color: #2e7d32; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #2e7d3280;">{relation.get("procedure", "")}</div>
                        <div style="margin: 0 10px;">➡️</div>
                        <div style="background-color: #c6282820; color: #c62828; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #c6282880;">{relation.get("for_condition", "")}</div>
                        <div style="margin-left: 10px; color: {confidence_color}; font-size: 0.8em;">({relation.get("confidence", "medium")} confidence)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display lab-condition relationships
            if "lab_condition_relations" in relationships and relationships["lab_condition_relations"]:
                st.markdown("#### 🧪➡️🩺 Lab Results associated with Conditions")
                for relation in relationships["lab_condition_relations"]:
                    confidence_color = get_confidence_color(relation.get("confidence", "medium"))
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="background-color: #6a1b9a20; color: #6a1b9a; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #6a1b9a80;">{relation.get("lab_result", "")}</div>
                        <div style="margin: 0 10px;">➡️</div>
                        <div style="background-color: #c6282820; color: #c62828; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #c6282880;">{relation.get("for_condition", "")}</div>
                        <div style="margin-left: 10px; color: {confidence_color}; font-size: 0.8em;">({relation.get("confidence", "medium")} confidence)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display provider-condition relationships
            if "provider_condition_relations" in relationships and relationships["provider_condition_relations"]:
                st.markdown("#### 👩‍⚕️➡️🩺 Providers treating Conditions")
                for relation in relationships["provider_condition_relations"]:
                    confidence_color = get_confidence_color(relation.get("confidence", "medium"))
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="background-color: #1a237e20; color: #1a237e; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #1a237e80;">{relation.get("provider", "")}</div>
                        <div style="margin: 0 10px;">➡️</div>
                        <div style="background-color: #c6282820; color: #c62828; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #c6282880;">{relation.get("treats_condition", "")}</div>
                        <div style="margin-left: 10px; color: {confidence_color}; font-size: 0.8em;">({relation.get("confidence", "medium")} confidence)</div>
                    </div>
                    """, unsafe_allow_html=True)

def display_category(category, entities, categories_config):
    """Helper function to display a category of medical entities"""
    if category not in entities or not entities[category]:
        return
    
    config = categories_config[category]
    entities_in_category = entities[category]
    count = len(entities_in_category)
    
    st.markdown(f"""
    <div style="padding: 10px; border: 1px solid {config['color']}; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="color: {config['color']}; margin-top: 0;">{config['icon']} {config['title']} ({count})</h4>
        <div style="max-height: 200px; overflow-y: auto;">
    """, unsafe_allow_html=True)
    
    for entity_data in entities_in_category:
        # Handle both string entities and dictionary entities
        if isinstance(entity_data, str):
            st.markdown(f"""
            <div style="background-color: {config['color']}10; padding: 5px 8px; margin-bottom: 5px; border-radius: 3px;">
                <span style="color: {config['color']}; font-weight: 500;">{entity_data}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display rich entity data with available fields
            entity_term = entity_data.get("term", "")
            
            # Start entity display
            st.markdown(f"""
            <div style="background-color: {config['color']}10; padding: 8px; margin-bottom: 8px; border-radius: 3px; border-left: 3px solid {config['color']};">
                <div style="color: {config['color']}; font-weight: 600; font-size: 1.05em;">{entity_term}</div>
            """, unsafe_allow_html=True)
            
            # Add normalized term if available and different
            if "normalized" in entity_data and entity_data["normalized"] != entity_term:
                st.markdown(f"""
                <div style="font-size: 0.9em; margin-top: 3px;">
                    <span style="color: #555; font-weight: 500;">Normalized:</span> {entity_data["normalized"]}
                </div>
                """, unsafe_allow_html=True)
            
            # Add other fields with appropriate formatting
            for key, value in entity_data.items():
                if key not in ["term", "normalized"] and value:
                    key_formatted = key.replace('_', ' ').title()
                    
                    # Format values differently based on the field type
                    value_formatted = value
                    if "date" in key.lower() and isinstance(value, str):
                        # Format dates with calendar icon
                        value_formatted = f'<span style="color: #666;">📅 {value}</span>'
                    elif "dosage" in key.lower() and isinstance(value, str):
                        # Format dosage with pill icon
                        value_formatted = f'<span style="color: #666;">💊 {value}</span>'
                    elif "frequency" in key.lower() and isinstance(value, str):
                        # Format frequency with clock icon
                        value_formatted = f'<span style="color: #666;">🕒 {value}</span>'
                    
                    st.markdown(f"""
                    <div style="font-size: 0.9em; margin-top: 2px;">
                        <span style="color: #555; font-weight: 500;">{key_formatted}:</span> {value_formatted}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Close the entity display
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Close the category container
    st.markdown("</div></div>", unsafe_allow_html=True)

def create_entity_network_graph(entities, relationships):
    """
    Creates a Pyvis network graph visualization of entity relationships
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes for entities
    added_nodes = set()
    
    # Add nodes for medications
    for entity in entities.get("medications", []):
        if isinstance(entity, str):
            node_id = entity
        else:
            node_id = entity.get("term", "")
        
        if node_id and node_id not in added_nodes:
            G.add_node(node_id, color="#0277bd", title=node_id, group="medications")
            added_nodes.add(node_id)
    
    # Add nodes for conditions
    for entity in entities.get("medical_conditions", []):
        if isinstance(entity, str):
            node_id = entity
        else:
            node_id = entity.get("term", "")
        
        if node_id and node_id not in added_nodes:
            G.add_node(node_id, color="#c62828", title=node_id, group="medical_conditions")
            added_nodes.add(node_id)
    
    # Add nodes for procedures
    for entity in entities.get("procedures", []):
        if isinstance(entity, str):
            node_id = entity
        else:
            node_id = entity.get("term", "")
        
        if node_id and node_id not in added_nodes:
            G.add_node(node_id, color="#2e7d32", title=node_id, group="procedures")
            added_nodes.add(node_id)
    
    # Add edges for relationships
    if relationships:
        # Add medication-condition relationships
        for relation in relationships.get("medication_condition_relations", []):
            med = relation.get("medication", "")
            cond = relation.get("treats_condition", "")
            if med and cond and med in added_nodes and cond in added_nodes:
                G.add_edge(med, cond, title="treats", color="#0277bd")
        
        # Add procedure-condition relationships
        for relation in relationships.get("procedure_condition_relations", []):
            proc = relation.get("procedure", "")
            cond = relation.get("for_condition", "")
            if proc and cond and proc in added_nodes and cond in added_nodes:
                G.add_edge(proc, cond, title="treats", color="#2e7d32")
    
    # Convert to Pyvis network
    nt = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # From NetworkX
    nt.from_nx(G)
    
    # Set options
    nt.set_options("""
    const options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        nt.save_graph(tmpfile.name)
        return tmpfile.name
# def create_entity_network_graph(entities, relationships):
#     """
#     Creates an enhanced Pyvis network graph visualization of entity relationships
#     """
#     # Create a NetworkX graph
#     G = nx.Graph()
    
#     # Define styling for entity types
#     entity_styles = {
#         "medications": {"color": "#0277bd", "shape": "dot", "size": 20, "group": "Medications"},
#         "medical_conditions": {"color": "#c62828", "shape": "square", "size": 20, "group": "Conditions"},
#         "procedures": {"color": "#2e7d32", "shape": "triangle", "size": 20, "group": "Procedures"},
#         "lab_results": {"color": "#6a1b9a", "shape": "diamond", "size": 15, "group": "Lab Results"},
#         "vital_signs": {"color": "#ef6c00", "shape": "star", "size": 15, "group": "Vital Signs"},
#         "anatomical_structures": {"color": "#3e2723", "shape": "ellipse", "size": 15, "group": "Anatomy"},
#         "medical_devices": {"color": "#546e7a", "shape": "hexagon", "size": 15, "group": "Devices"},
#         "healthcare_providers": {"color": "#1a237e", "shape": "box", "size": 20, "group": "Providers"}
#     }

#     # Add nodes for entities with detailed metadata
#     added_nodes = set()
#     for category, style in entity_styles.items():
#         for entity in entities.get(category, []):
#             if isinstance(entity, str):
#                 node_id = entity
#                 title = f"{category.replace('_', ' ').title()}: {entity}"
#             else:
#                 node_id = entity.get("term", "")
#                 # Build detailed tooltip
#                 details = [f"{key.replace('_', ' ').title()}: {value}" for key, value in entity.items() if value]
#                 title = f"{category.replace('_', ' ').title()}: {node_id}\n" + "\n".join(details)
            
#             if node_id and node_id not in added_nodes:
#                 G.add_node(
#                     node_id,
#                     label=node_id,
#                     title=title,
#                     color=style["color"],
#                     shape=style["shape"],
#                     size=style["size"],
#                     group=style["group"]
#                 )
#                 added_nodes.add(node_id)

#     # Calculate node sizes based on degree (optional)
#     degree_dict = dict(G.degree())
#     for node in G.nodes():
#         G.nodes[node]["size"] = max(15, min(40, degree_dict[node] * 5))  # Scale size between 15 and 40

#     # Add edges for relationships with labels
#     if relationships:
#         for relation in relationships.get("medication_condition_relations", []):
#             med = relation.get("medication", "")
#             cond = relation.get("treats_condition", "")
#             conf = relation.get("confidence", "medium")
#             if med and cond and med in added_nodes and cond in added_nodes:
#                 G.add_edge(
#                     med, cond,
#                     title=f"Treats (Confidence: {conf})",
#                     color="#0277bd",
#                     width=2,
#                     label="treats"
#                 )
        
#         for relation in relationships.get("procedure_condition_relations", []):
#             proc = relation.get("procedure", "")
#             cond = relation.get("for_condition", "")
#             conf = relation.get("confidence", "medium")
#             if proc and cond and proc in added_nodes and cond in added_nodes:
#                 G.add_edge(
#                     proc, cond,
#                     title=f"For Condition (Confidence: {conf})",
#                     color="#2e7d32",
#                     width=2,
#                     label="for"
#                 )
        
#         for relation in relationships.get("lab_condition_relations", []):
#             lab = relation.get("lab_result", "")
#             cond = relation.get("for_condition", "")
#             conf = relation.get("confidence", "medium")
#             if lab and cond and lab in added_nodes and cond in added_nodes:
#                 G.add_edge(
#                     lab, cond,
#                     title=f"Associated (Confidence: {conf})",
#                     color="#6a1b9a",
#                     width=2,
#                     label="associated"
#                 )
        
#         for relation in relationships.get("provider_condition_relations", []):
#             prov = relation.get("provider", "")
#             cond = relation.get("treats_condition", "")
#             conf = relation.get("confidence", "medium")
#             if prov and cond and prov in added_nodes and cond in added_nodes:
#                 G.add_edge(
#                     prov, cond,
#                     title=f"Treats (Confidence: {conf})",
#                     color="#1a237e",
#                     width=2,
#                     label="treats"
#                 )

#     # Initialize Pyvis network
#     nt = Network(
#         height="600px",
#         width="100%",
#         bgcolor="#f5f5f5",  # Light gray background
#         font_color="#333333",  # Darker font for contrast
#         directed=False
#     )
    
#     # Convert from NetworkX
#     nt.from_nx(G)
    
#     # Enhanced physics and styling options
#     nt.set_options("""
#     {
#       "nodes": {
#         "font": {
#           "size": 14,
#           "face": "Arial",
#           "color": "#333333"
#         },
#         "borderWidth": 2
#       },
#       "edges": {
#         "color": {
#           "inherit": false
#         },
#         "smooth": {
#           "type": "continuous"
#         },
#         "font": {
#           "size": 12,
#           "align": "middle"
#         },
#         "width": 2
#       },
#       "physics": {
#         "forceAtlas2Based": {
#           "gravitationalConstant": -100,
#           "centralGravity": 0.005,
#           "springLength": 150,
#           "springConstant": 0.1,
#           "avoidOverlap": 1
#         },
#         "maxVelocity": 50,
#         "minVelocity": 0.1,
#         "solver": "forceAtlas2Based"
#       },
#       "interaction": {
#         "hover": true,
#         "tooltipDelay": 200,
#         "dragNodes": true,
#         "dragView": true,
#         "zoomView": true
#       }
#     }
#     """)
    
#     # Add buttons for interactivity
#     nt.show_buttons(filter_=['physics', 'nodes', 'edges'])  # Show configuration panel
    
#     # Generate HTML file
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
#         nt.save_graph(tmpfile.name)
#         return tmpfile.name

def generate_summary(text):
    """
    Generates a structured medical summary of the medical record text using Gemini
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Generate a structured medical summary from the following medical record text.
    Follow this exact format and include all relevant details:
    
    ## Patient Overview
    - Age/Sex: [age, gender]
    - Key Demographics: [occupation, residence if relevant]
    
    ## Active Medical History
    - Chronic Conditions: [list with diagnosis dates]
    - Previous Hospitalizations: [dates/reasons]
    - Surgeries/Procedures: [list with dates]
    
    ## Presenting Complaint
    - Primary Reason: [chief complaint]
    - Onset: [when symptoms started]
    - Progression: [course of symptoms]
    
    ## Clinical Findings
    - Vital Signs: [BP, HR, Temp, etc.]
    - Physical Exam: [key findings]
    - Cognitive Status: [MMSE score if available]
    
    ## Diagnostic Results
    - Labs: [significant results with dates]
    - Imaging: [key findings with dates]
    - Other Tests: [ECG, biopsies, etc.]
    
    ## Current Treatment
    - Medications: [list with dosages]
    - Therapies: [ongoing treatments]
    - Care Team: [managing providers]
    
    ## Assessment & Plan
    - Working Diagnoses: [confirmed/suspected]
    - Management Strategy: [next steps]
    - Follow-up: [scheduled appointments]
    
    ## Special Considerations
    - Allergies: [list if any]
    - Social Factors: [family support, living situation]
    
    Use concise medical terminology. Include relevant dates and metrics.
    Maintain original lab values and medication dosages from the text.
    
    Medical Record Text:
    {text}
    """

    response = model.generate_content(prompt)
    summary = response.text.strip()
    
    # Clean up any markdown formatting
    summary = re.sub(r"\*\*", "", summary)  # Remove bold markers
    return summary


def user_input(question):
    """
    Processes user questions about the medical record and returns relevant answers
    """
    try:
        # Load vector store if it exists
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Modified line: Add allow_dangerous_deserialization
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True  # Safety measure for trusted sources
        )
        
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
    
def display_color_legend():
    """Displays an enhanced legend for entity color codes"""
    st.sidebar.markdown("### Entity Color Legend")
    
    legend_html = """
    <div style="padding: 10px; background-color: #000000; border-radius: 5px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #0277bd; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">💊 Medications</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #c62828; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">🩺 Medical Conditions</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #2e7d32; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">⚕️ Procedures</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #6a1b9a; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">🧪 Lab Results</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #ef6c00; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">📊 Vital Signs</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #3e2723; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">🫀 Anatomical Structures</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: #546e7a; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">🔧 Medical Devices</div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background-color: #1a237e; margin-right: 10px; border-radius: 3px;"></div>
            <div style="font-weight: 500;">👩‍⚕️ Healthcare Providers</div>
        </div>
    </div>
    """
    
    st.sidebar.markdown(legend_html, unsafe_allow_html=True)

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

def display_timeline(timeline_data):
    """
    Displays a timeline visualization of medical events
    """
    if not timeline_data:
        st.info("No temporal data available for timeline visualization.")
        return
    
    # Sort timeline data by date
    timeline_data.sort(key=lambda x: x["start_date"])
    
    # Create a timeline visualization
    st.markdown("### Medical Timeline")
    
    # Create the timeline HTML
    timeline_html = """
    <div style="width: 100%; overflow-x: auto;">
        <div style="display: flex; align-items: center; padding: 10px 0;">
            <!-- Timeline axis -->
            <div style="width: 100%; height: 4px; background-color: #ccc; position: relative; margin: 20px 0;">
    """
    
    # Add timeline events
    for i, event in enumerate(timeline_data):
        # Calculate position (simple for now, can be improved)
        position = i * (100 / max(len(timeline_data), 1))
        position = min(max(position, 5), 95)  # Keep within 5-95% range
        
        # Event type icon
        icon = "💊" if event["type"] == "medication" else "🩺" if event["type"] == "condition" else "⚕️"
        
        # Create event marker
        timeline_html += f"""
        <div style="position: absolute; left: {position}%; transform: translateX(-50%);">
            <div style="width: 16px; height: 16px; background-color: {event["color"]}; border-radius: 50%; margin-bottom: 8px;"></div>
            <div style="position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%); background-color: {event["color"]}20; border: 1px solid {event["color"]}; border-radius: 4px; padding: 4px 8px; white-space: nowrap;">
                <div style="font-weight: bold; color: {event["color"]};">{icon} {event["name"]}</div>
                <div style="font-size: 0.8em;">{event["start_date"]}</div>
            </div>
        </div>
        """
    
    # Close the timeline HTML
    timeline_html += """
            </div>
        </div>
    </div>
    """
    
    # Display the timeline
    st.markdown(timeline_html, unsafe_allow_html=True)

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

# ------------------- Updated Streamlit UI -------------------

def main():
    """Main function to run Streamlit app with enhanced NER"""
    st.title("📄 Medical Record AI Assistant")
    st.write("Upload and analyze your medical records. The AI will assist you in understanding your health data.")

    # Display the enhanced color legend in the sidebar
    display_color_legend()
    
    # Add stats summary to sidebar
    if 'entities' in st.session_state and st.session_state.entities:
        st.sidebar.markdown("### Document Statistics")
        
        total_entities = sum(len(entities) for entities in st.session_state.entities.values())
        
        stats_html = f"""
        <div style="padding: 10px; background-color: #000000; border-radius: 5px; margin-bottom: 15px;">
            <div style="font-weight: 600; font-size: 1.1em; margin-bottom: 8px;">Entity Summary</div>
            <div>Total Medical Entities: <span style="font-weight: 600;">{total_entities}</span></div>
        """
        
        # Add counts for each category
        for category, items in st.session_state.entities.items():
            if items:
                category_name = category.replace('_', ' ').title()
                stats_html += f'<div>{category_name}: <span style="font-weight: 600;">{len(items)}</span></div>'
        
        stats_html += "</div>"
        st.sidebar.markdown(stats_html, unsafe_allow_html=True)

    tabs = st.tabs(["Text Summary", "Medical Entities", "Entity Relationships", "Conversational AI"])

    # Initialize session state for storing extracted data
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'entities' not in st.session_state:
        st.session_state.entities = {}
    if 'relationships' not in st.session_state:
        st.session_state.relationships = {}
    if 'timeline_data' not in st.session_state:
        st.session_state.timeline_data = []

    with tabs[0]:
        st.header("Text Summary")
        pdf_docs = st.file_uploader("📂 Upload your medical records (PDF)", type=["pdf"], accept_multiple_files=True, key="summary_uploader")
        
        if pdf_docs:
            with st.spinner("🔄 Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text
                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Documents processed successfully. Generating summary...")
                
                summary = generate_summary(raw_text)
                entities, relationships, timeline_data = ner_processing(raw_text)
                
                st.session_state.entities = entities
                st.session_state.relationships = relationships
                st.session_state.timeline_data = timeline_data
                
                highlighted_summary = highlight_entities(summary, entities)
                
                st.markdown("### Summary of Medical Records")
                st.markdown(highlighted_summary, unsafe_allow_html=True)

    with tabs[1]:
        st.header("Medical Entities")
        
        pdf_docs = st.file_uploader("📂 Upload your records (PDF)", type=["pdf"], accept_multiple_files=True, key="ner_uploader")
        
        if pdf_docs:
            with st.spinner("🔄 Processing documents for entity extraction..."):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text
                
                entities, relationships, timeline_data = ner_processing(raw_text)
                
                st.session_state.entities = entities
                st.session_state.relationships = relationships
                st.session_state.timeline_data = timeline_data
                
                # Display entity lists in a structured format
                st.markdown("### Extracted Medical Entities")
                display_entity_lists(entities)
                
                # Add a divider
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Add a toggle for the full text
                with st.expander("View Full Text with Highlighted Entities"):
                    highlighted_text = highlight_entities(raw_text, entities)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
        
        elif 'entities' in st.session_state and st.session_state.entities:
            # Display entity lists from session state
            st.markdown("### Extracted Medical Entities")
            display_entity_lists(st.session_state.entities)
            
            # Add a divider
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Add a toggle for the full text
            with st.expander("View Full Text with Highlighted Entities"):
                highlighted_text = highlight_entities(st.session_state.raw_text, st.session_state.entities)
                st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            st.info("Please upload documents to extract medical entities.")

    with tabs[2]:
        st.header("Entity Relationships")
        
        if 'relationships' in st.session_state and st.session_state.relationships and 'entities' in st.session_state and st.session_state.entities:
            # Display relationship information
            relationships = st.session_state.relationships
            entities = st.session_state.entities
            
            # Create tabs for different ways to view relationships
            relationship_view_tabs = st.tabs(["List View", "Network View"])
            
            with relationship_view_tabs[0]:
                # List view of relationships
                if relationships:
                    # Display entity relationships
                    st.markdown("### Entity Relationships")
                    
                    # Medication-condition relationships
                    if "medication_condition_relations" in relationships and relationships["medication_condition_relations"]:
                        st.markdown("#### 💊➡️🩺 Medications treating Conditions")
                        for relation in relationships["medication_condition_relations"]:
                            confidence_color = get_confidence_color(relation.get("confidence", "medium"))
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <div style="background-color: #0277bd20; color: #0277bd; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #0277bd80;">{relation.get("medication", "")}</div>
                                <div style="margin: 0 10px;">➡️</div>
                                <div style="background-color: #c6282820; color: #c62828; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid #c6282880;">{relation.get("treats_condition", "")}</div>
                                <div style="margin-left: 10px; color: {confidence_color}; font-size: 0.8em;">({relation.get("confidence", "medium")} confidence)</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add other relationship types as needed
                    for rel_type in ["procedure_condition_relations", "lab_condition_relations", "provider_condition_relations"]:
                        if rel_type in relationships and relationships[rel_type]:
                            if rel_type == "procedure_condition_relations":
                                st.markdown("#### ⚕️➡️🩺 Procedures for Conditions")
                                key1, key2 = "procedure", "for_condition"
                                color1, color2 = "#2e7d32", "#c62828"
                            elif rel_type == "lab_condition_relations":
                                st.markdown("#### 🧪➡️🩺 Lab Results associated with Conditions")
                                key1, key2 = "lab_result", "associated_condition"
                                color1, color2 = "#6a1b9a", "#c62828"
                            elif rel_type == "provider_condition_relations":
                                st.markdown("#### 👩‍⚕️➡️🩺 Providers treating Conditions")
                                key1, key2 = "provider", "treats_condition"
                                color1, color2 = "#1a237e", "#c62828"
                            
                            for relation in relationships[rel_type]:
                                confidence_color = get_confidence_color(relation.get("confidence", "medium"))
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                    <div style="background-color: {color1}20; color: {color1}; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid {color1}80;">{relation.get(key1, "")}</div>
                                    <div style="margin: 0 10px;">➡️</div>
                                    <div style="background-color: {color2}20; color: {color2}; font-weight: bold; padding: 3px 8px; border-radius: 3px; border: 1px solid {color2}80;">{relation.get(key2, "")}</div>
                                    <div style="margin-left: 10px; color: {confidence_color}; font-size: 0.8em;">({relation.get("confidence", "medium")} confidence)</div>
                                </div>
                                """, unsafe_allow_html=True)
            
            with relationship_view_tabs[1]:
                # Network view of relationships
                st.markdown("### Entity Network Graph")
                st.info("This network graph shows relationships between medical entities.")
                
                try:
                    # Install required packages if needed
                    import importlib
                    if importlib.util.find_spec("pyvis") is None:
                        st.warning("Network visualization requires PyVis package. Please install it with 'pip install pyvis networkx' and restart the application.")
                    else:
                        html_file = create_entity_network_graph(entities, relationships)
                        with open(html_file, 'r', encoding='utf-8') as f:
                            html_data = f.read()

                        import streamlit.components.v1 as components
                        components.html(html_data, height=600)
                except Exception as e:
                    st.error(f"Could not generate network graph: {str(e)}")
                    st.info("To enable network visualization, ensure you have pyvis and networkx installed: 'pip install pyvis networkx'")
        else:
            st.info("Please upload documents to analyze entity relationships.")


    with tabs[3]:
        st.header("Conversational AI")
        user_question = st.text_input("💬 Ask a question about the records:")
        
        if user_question and st.session_state.raw_text:
            with st.spinner("🔍 Searching for answers..."):
                response, new_entities = user_input(user_question)
                st.markdown(response, unsafe_allow_html=True)
                if new_entities:
                    with st.expander("Entities in Response"):
                        display_entity_lists(new_entities)
        elif user_question:
            st.warning("⚠️ Please upload documents in the 'Text Summary' or 'Medical Entities' tab first.")

if __name__ == "__main__":
    main()
