# Import necessary modules
from flask import Flask, redirect, url_for, render_template, request, session
import uuid
import time
from functools import wraps
import logging
import os
import re  # Make sure this import is here
import ast  # Add this for ast.literal_eval
import json  # Add this for JSON handling

# Import all functions from llm_functions
from llm_functions import (
    initialize_conversation,
    initialize_conv_reco,
    get_chat_model_completions,
    moderation_check,
    intent_confirmation_layer,
    compare_laptops_with_user,
    recommendation_validation,
    get_user_requirement_string,
    get_chat_completions_func_calling,
    dictionary_present,
    LAPTOP_DATASET  # Import the dataset directly
)

# Add this near the top of your file
DEBUG_MODE = True  # Set to False in production

# Check if API key file exists
try:
    with open("OpenAI_API_Key.txt", 'r') as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError("API key file is empty")
    
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading OpenAI API key: {e}")
    print("Please ensure 'OpenAI_API_Key.txt' exists and contains a valid API key")
    exit(1)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure random key in production

# Use a dictionary to store conversations for different sessions
conversations = {}
recommendations = {}

# Set session lifetime to 1 hour (3600 seconds)
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

@app.before_request
def make_session_permanent():
    session.permanent = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shopai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple rate limiting decorator
def rate_limit(limit=15, per=60):
    last_requests = {}
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Skip rate limiting in debug mode
            if DEBUG_MODE:
                return f(*args, **kwargs)
                
            now = time.time()
            
            # Create a new session if one doesn't exist
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            
            session_id = session.get('session_id', 'default')
            
            if session_id not in last_requests:
                last_requests[session_id] = []
            
            # Clean old requests
            last_requests[session_id] = [t for t in last_requests[session_id] if now - t < per]
            
            if len(last_requests[session_id]) >= limit:
                return render_template("conversation_bot.html", 
                                      name_xyz=[{"bot": "You've sent too many requests. Please wait a moment before trying again."}])
            
            last_requests[session_id].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.route("/")
def default_func():
    # Create a new session if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    # Initialize conversation for this session if it doesn't exist
    if session_id not in conversations:
        conversations[session_id] = initialize_conversation()
        recommendations[session_id] = None
    
    # Get conversation history for this session
    conversation_bot = []
    # Skip the system message (index 0) when creating the display
    for message in conversations[session_id][1:]:
        if message["role"] == "user":
            conversation_bot.append({"user": message["content"]})
        elif message["role"] == "assistant":
            conversation_bot.append({"bot": message["content"]})
    
    # If no messages to display yet, add the welcome message
    if not conversation_bot and len(conversations[session_id]) > 1:
        welcome_message = conversations[session_id][1]["content"]
        conversation_bot.append({"bot": welcome_message})
    
    return render_template("conversation_bot.html", name_xyz=conversation_bot)

@app.route("/end_conversation", methods = ['POST','GET'])
def end_conv():
    global conversations, recommendations
    
    # Create a new session if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session.get('session_id')
    
    # Initialize conversation for this session
    conversations[session_id] = initialize_conversation()
    recommendations[session_id] = None
    
    return redirect(url_for('default_func'))

@app.route("/conversation", methods=['POST'])
@rate_limit(limit=15, per=60)  # 15 requests per minute instead of 5
def invite():
    global conversations, recommendations
    
    # Create a new session if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session.get('session_id')
    
    # Initialize conversation for this session if it doesn't exist
    if session_id not in conversations:
        conversations[session_id] = initialize_conversation()
        recommendations[session_id] = None
    
    try:
        # Debug information
        logger.info(f"Form data keys: {list(request.form.keys())}")
        
        # Get user input from form
        user_input = ""
        if request.form and "user_input_message" in request.form:
            user_input = request.form["user_input_message"]
        
        if not user_input:
            logger.error("Empty or missing user input")
            return render_template("conversation_bot.html", 
                                  name_xyz=[{"bot": "Sorry, I couldn't understand your message. Please try again with a clear question or request."}])
        
        logger.info(f"Received user input: {user_input}")
        
        # Re-enable moderation check
        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))
        
        # Check for dataset-specific queries
        lower_input = user_input.lower()
        
        # Expanded inventory-related keywords
        inventory_keywords = [
            "inventory", "stock", "available", "have", "offer", "sell", "selection",
            "catalog", "collection", "range", "options", "models", "products", "laptops you have"
        ]
        
        # Feature-specific keywords and patterns
        storage_pattern = re.search(r'(\d+)\s*(tb|gb)\s+storage', lower_input)
        ram_pattern = re.search(r'(\d+)\s*gb\s+ram', lower_input)
        processor_keywords = ["i3", "i5", "i7", "i9", "ryzen", "intel", "amd"]
        display_keywords = ["4k", "fhd", "qhd", "uhd", "oled", "touch", "display"]
        
        # Additional feature patterns
        battery_keywords = ["battery", "battery life", "long lasting", "hours of battery"]
        weight_keywords = ["weight", "lightweight", "light", "portable", "heavy"]
        graphics_keywords = ["gpu", "graphics", "rtx", "gtx", "geforce", "radeon", "gaming"]
        
        # Comparison patterns
        comparison_pattern = re.search(r'(compare|vs|versus|better|difference between)\s+([a-zA-Z0-9\s]+)\s+and\s+([a-zA-Z0-9\s]+)', lower_input)
        
        # Check if this is a feature-specific query
        is_feature_query = (
            storage_pattern or 
            ram_pattern or 
            any(keyword in lower_input for keyword in processor_keywords) or
            any(keyword in lower_input for keyword in display_keywords) or
            any(keyword in lower_input for keyword in battery_keywords) or
            any(keyword in lower_input for keyword in weight_keywords) or
            any(keyword in lower_input for keyword in graphics_keywords)
        )
        
        # Check if this is a comparison query
        is_comparison_query = comparison_pattern is not None
        
        # Handle inventory-related, feature-specific, or comparison queries
        if any(keyword in lower_input for keyword in inventory_keywords) or is_feature_query or is_comparison_query:
            # No need to import LAPTOP_DATASET here since we imported it at the top
            
            # Handle comparison queries
            if is_comparison_query:
                item1 = comparison_pattern.group(2).strip().lower()
                item2 = comparison_pattern.group(3).strip().lower()
                
                # Try to find the laptops being compared
                laptop1_matches = []
                laptop2_matches = []
                
                # Fuzzy matching for laptop models
                for laptop in LAPTOP_DATASET:
                    brand_model = f"{laptop.get('Brand', '').lower()} {laptop.get('Model', '').lower()}"
                    
                    # Check for partial matches in brand+model
                    if item1 in brand_model or any(word in brand_model for word in item1.split()):
                        laptop1_matches.append(laptop)
                    
                    if item2 in brand_model or any(word in brand_model for word in item2.split()):
                        laptop2_matches.append(laptop)
                
                # If we found matches for both laptops
                if laptop1_matches and laptop2_matches:
                    laptop1 = laptop1_matches[0]  # Take the first match
                    laptop2 = laptop2_matches[0]  # Take the first match
                    
                    # Create a comparison table
                    comparison = f"Here's a comparison between {laptop1['Brand']} {laptop1['Model']} and {laptop2['Brand']} {laptop2['Model']}:\n\n"
                    comparison += f"| Feature | {laptop1['Brand']} {laptop1['Model']} | {laptop2['Brand']} {laptop2['Model']} |\n"
                    comparison += f"|---------|--------------------|-----------------|\n"
                    comparison += f"| Processor | {laptop1['Processor']} | {laptop2['Processor']} |\n"
                    comparison += f"| RAM | {laptop1['RAM']} | {laptop2['RAM']} |\n"
                    comparison += f"| Storage | {laptop1['Storage']} | {laptop2['Storage']} |\n"
                    comparison += f"| Display | {laptop1['Display']} | {laptop2['Display']} |\n"
                    comparison += f"| Price | ₹{laptop1['Price']} | ₹{laptop2['Price']} |\n"
                    
                    # Add performance metrics if available
                    if 'GPU' in laptop1 and 'GPU' in laptop2:
                        comparison += f"| GPU Performance | {laptop1['GPU']} | {laptop2['GPU']} |\n"
                    if 'Processing_Speed' in laptop1 and 'Processing_Speed' in laptop2:
                        comparison += f"| Processing Speed | {laptop1['Processing_Speed']} | {laptop2['Processing_Speed']} |\n"
                    if 'Multitasking' in laptop1 and 'Multitasking' in laptop2:
                        comparison += f"| Multitasking | {laptop1['Multitasking']} | {laptop2['Multitasking']} |\n"
                    
                    # Add a summary
                    comparison += f"\nSummary:\n"
                    
                    # Compare processors
                    if 'i9' in laptop1['Processor'].lower() and 'i7' in laptop2['Processor'].lower():
                        comparison += f"- The {laptop1['Brand']} {laptop1['Model']} has a more powerful processor.\n"
                    elif 'i7' in laptop1['Processor'].lower() and 'i5' in laptop2['Processor'].lower():
                        comparison += f"- The {laptop1['Brand']} {laptop1['Model']} has a more powerful processor.\n"
                    elif 'i9' in laptop2['Processor'].lower() and 'i7' in laptop1['Processor'].lower():
                        comparison += f"- The {laptop2['Brand']} {laptop2['Model']} has a more powerful processor.\n"
                    elif 'i7' in laptop2['Processor'].lower() and 'i5' in laptop1['Processor'].lower():
                        comparison += f"- The {laptop2['Brand']} {laptop2['Model']} has a more powerful processor.\n"
                    
                    # Compare RAM
                    ram1 = int(laptop1['RAM'].lower().replace('gb', '').strip())
                    ram2 = int(laptop2['RAM'].lower().replace('gb', '').strip())
                    if ram1 > ram2:
                        comparison += f"- The {laptop1['Brand']} {laptop1['Model']} has more RAM ({laptop1['RAM']} vs {laptop2['RAM']}).\n"
                    elif ram2 > ram1:
                        comparison += f"- The {laptop2['Brand']} {laptop2['Model']} has more RAM ({laptop2['RAM']} vs {laptop1['RAM']}).\n"
                    
                    # Compare price
                    if laptop1['Price'] < laptop2['Price']:
                        comparison += f"- The {laptop1['Brand']} {laptop1['Model']} is more affordable (₹{laptop1['Price']} vs ₹{laptop2['Price']}).\n"
                    elif laptop2['Price'] < laptop1['Price']:
                        comparison += f"- The {laptop2['Brand']} {laptop2['Model']} is more affordable (₹{laptop2['Price']} vs ₹{laptop1['Price']}).\n"
                    
                    # Add recommendation based on use case if mentioned in the query
                    if "gaming" in lower_input:
                        if laptop1.get('GPU', 'Medium').lower() == 'high' and laptop2.get('GPU', 'Medium').lower() != 'high':
                            comparison += f"\nFor gaming, the {laptop1['Brand']} {laptop1['Model']} would be a better choice due to its superior GPU performance."
                        elif laptop2.get('GPU', 'Medium').lower() == 'high' and laptop1.get('GPU', 'Medium').lower() != 'high':
                            comparison += f"\nFor gaming, the {laptop2['Brand']} {laptop2['Model']} would be a better choice due to its superior GPU performance."
                    
                    if "programming" in lower_input or "coding" in lower_input:
                        if ram1 >= 16 and ram2 < 16:
                            comparison += f"\nFor programming, the {laptop1['Brand']} {laptop1['Model']} would be a better choice due to its higher RAM capacity."
                        elif ram2 >= 16 and ram1 < 16:
                            comparison += f"\nFor programming, the {laptop2['Brand']} {laptop2['Model']} would be a better choice due to its higher RAM capacity."
                    
                    response = comparison
                    
                else:
                    # If we couldn't find one or both laptops
                    if not laptop1_matches and not laptop2_matches:
                        response = f"I couldn't find information about either '{item1}' or '{item2}' in our dataset. Could you provide more specific model names?"
                    elif not laptop1_matches:
                        response = f"I couldn't find information about '{item1}' in our dataset. Could you provide a more specific model name?"
                    else:
                        response = f"I couldn't find information about '{item2}' in our dataset. Could you provide a more specific model name?"
            else:
                # Extract brand if mentioned
                brands = ["dell", "hp", "lenovo", "asus", "acer", "microsoft", "apple", "msi", "gigabyte", "razer"]
                mentioned_brand = next((brand for brand in brands if brand in lower_input), None)
                
                # Extract price constraints if mentioned
                price_under_match = re.search(r'under\s+[₹$]?(\d{4,6})', lower_input)
                price_under = int(price_under_match.group(1)) if price_under_match else None
                
                # Filter laptops based on criteria
                filtered_laptops = LAPTOP_DATASET
                
                # Apply brand filter if mentioned
                if mentioned_brand:
                    filtered_laptops = [laptop for laptop in filtered_laptops 
                                       if mentioned_brand in laptop.get('Brand', '').lower()]
                
                # Apply price filter if mentioned
                if price_under:
                    filtered_laptops = [laptop for laptop in filtered_laptops 
                                       if laptop.get('Price', float('inf')) < price_under]
                
                # Apply storage filter if mentioned
                if storage_pattern:
                    storage_size = int(storage_pattern.group(1))
                    storage_unit = storage_pattern.group(2).lower()
                    
                    # Convert to GB for comparison
                    if storage_unit == 'tb':
                        storage_size *= 1000
                    
                    # Filter laptops by storage
                    filtered_laptops = [laptop for laptop in filtered_laptops 
                                       if str(storage_size) in laptop.get('Storage', '').lower() or
                                       (storage_unit == 'tb' and '1tb' in laptop.get('Storage', '').lower())]
                
                # Apply RAM filter if mentioned
                if ram_pattern:
                    ram_size = ram_pattern.group(1)
                    filtered_laptops = [laptop for laptop in filtered_laptops 
                                       if ram_size in laptop.get('RAM', '').lower()]
                
                # Apply processor filter if mentioned
                for keyword in processor_keywords:
                    if keyword in lower_input:
                        filtered_laptops = [laptop for laptop in filtered_laptops 
                                           if keyword in laptop.get('Processor', '').lower()]
                
                # Apply display filter if mentioned
                for keyword in display_keywords:
                    if keyword in lower_input:
                        filtered_laptops = [laptop for laptop in filtered_laptops 
                                           if keyword in laptop.get('Display', '').lower()]
                
                # Apply battery life filter if mentioned
                if any(keyword in lower_input for keyword in battery_keywords):
                    # Since we don't have actual battery life data, use a proxy like portability
                    filtered_laptops = [laptop for laptop in filtered_laptops 
                                       if laptop.get('Portability', 'Medium').lower() in ['high', 'medium']]
                
                # Apply weight/portability filter if mentioned
                if any(keyword in lower_input for keyword in weight_keywords):
                    filtered_laptops = [laptop for laptop in filtered_laptops 
                                       if laptop.get('Portability', 'Medium').lower() == 'high']
                
                # Apply graphics/GPU filter if mentioned
                if any(keyword in lower_input for keyword in graphics_keywords):
                    # If no laptops match all criteria but it's a gaming query, show gaming laptops anyway
                    gaming_laptops = [laptop for laptop in LAPTOP_DATASET 
                                     if laptop.get('GPU', 'Medium').lower() == 'high']
                    
                    # If we have gaming laptops, show them even if they don't match all criteria
                    if gaming_laptops:
                        laptop_info = "\n".join([
                            f"- {laptop['Brand']} {laptop['Model']}: {laptop['Processor']}, {laptop['RAM']}, {laptop['Storage']}, {laptop['Display']}, ₹{laptop['Price']}"
                            for laptop in gaming_laptops[:5]
                        ])
                        
                        response = f"I don't have any laptops with high-performance graphics that match all your criteria. However, here are some gaming laptops from our dataset:\n\n{laptop_info}\n\nWould you like more information about any of these models?"
                    else:
                        response = f"I don't have any laptops with high-performance graphics in our dataset. Would you like to see our best performance laptops instead?"
                
                # Limit to top 5 laptops to avoid overwhelming responses
                sample_laptops = filtered_laptops[:5]
                
                if sample_laptops:
                    laptop_info = "\n".join([
                        f"- {laptop['Brand']} {laptop['Model']}: {laptop['Processor']}, {laptop['RAM']}, {laptop['Storage']}, {laptop['Display']}, ₹{laptop['Price']}"
                        for laptop in sample_laptops
                    ])
                    
                    total_count = len(filtered_laptops)
                    shown_count = len(sample_laptops)
                    
                    # Craft response based on the query type
                    if storage_pattern:
                        storage_desc = f"{storage_pattern.group(1)}{storage_pattern.group(2).upper()}"
                        if shown_count < total_count:
                            response = f"Yes, we have {total_count} laptops with {storage_desc} storage in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"Yes, we have the following laptops with {storage_desc} storage in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    elif ram_pattern:
                        if shown_count < total_count:
                            response = f"Yes, we have {total_count} laptops with {ram_pattern.group(1)}GB RAM in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"Yes, we have the following laptops with {ram_pattern.group(1)}GB RAM in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    elif any(keyword in lower_input for keyword in battery_keywords):
                        if shown_count < total_count:
                            response = f"We have {total_count} laptops with good battery life in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"We have the following laptops with good battery life in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    elif any(keyword in lower_input for keyword in weight_keywords):
                        if shown_count < total_count:
                            response = f"We have {total_count} lightweight and portable laptops in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"We have the following lightweight and portable laptops in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    elif any(keyword in lower_input for keyword in graphics_keywords):
                        if shown_count < total_count:
                            response = f"We have {total_count} laptops with high-performance graphics in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"We have the following laptops with high-performance graphics in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    elif mentioned_brand:
                        brand_name = mentioned_brand.capitalize()
                        if shown_count < total_count:
                            response = f"Yes, we have {total_count} {brand_name} laptops in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"Yes, we have the following {brand_name} laptops in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    elif price_under:
                        if shown_count < total_count:
                            response = f"We have {total_count} laptops under ₹{price_under} in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                        else:
                            response = f"We have the following laptops under ₹{price_under} in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                    else:
                        if shown_count < total_count:
                            response = f"We have {total_count} laptops in our dataset. Here are {shown_count} examples:\n\n{laptop_info}\n\nWould you like more specific information or help narrowing down your options?"
                        else:
                            response = f"We have the following laptops in our dataset:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                else:
                    if storage_pattern:
                        storage_desc = f"{storage_pattern.group(1)}{storage_pattern.group(2).upper()}"
                        response = f"I don't have any laptops with {storage_desc} storage in our current dataset. Would you like to see other storage options?"
                    elif ram_pattern:
                        response = f"I don't have any laptops with {ram_pattern.group(1)}GB RAM in our current dataset. Would you like to see other RAM configurations?"
                    elif any(keyword in lower_input for keyword in battery_keywords):
                        response = f"I don't have specific battery life information in our dataset. Would you like to see laptops with high portability instead?"
                    elif any(keyword in lower_input for keyword in weight_keywords):
                        response = f"I don't have specific weight information in our dataset. Would you like to see laptops with high portability instead?"
                    elif any(keyword in lower_input for keyword in graphics_keywords):
                        response = f"I don't have any laptops with high-performance graphics in our dataset. Would you like to see our best performance laptops instead?"
                    elif mentioned_brand:
                        response = f"I don't have any {mentioned_brand.capitalize()} laptops in our current dataset. Would you like to see laptops from other brands?"
                    elif price_under:
                        response = f"I don't have any laptops under ₹{price_under} in our current dataset. The cheapest laptop we have is priced at ₹{min([laptop.get('Price', float('inf')) for laptop in LAPTOP_DATASET])}."
                    else:
                        response = f"We have {len(LAPTOP_DATASET)} laptops in our dataset. Could you provide more specific requirements so I can recommend the best options for you?"
            
            # Add user message to conversation history
            conversations[session_id].append({"role": "user", "content": user_input})
            # Add assistant response to conversation history
            conversations[session_id].append({"role": "assistant", "content": response})
            
            # Create conversation_bot for rendering
            conversation_bot = []
            for message in conversations[session_id][1:]:  # Skip system message
                if message["role"] == "user":
                    conversation_bot.append({"user": message["content"]})
                elif message["role"] == "assistant":
                    conversation_bot.append({"bot": message["content"]})
                    
            return render_template("conversation_bot.html", name_xyz=conversation_bot)
            
        # Special handler for comprehensive gaming requirements
        if "gaming" in lower_input and "gpu" in lower_input:
            # Extract budget if mentioned
            budget_pattern = re.search(r'budget\s+(?:is|of|around|about)?\s+[₹$]?(\d{4,6})', lower_input)
            budget = int(budget_pattern.group(1)) if budget_pattern else 100000  # Default high budget
            
            # Start with all laptops
            gaming_candidates = LAPTOP_DATASET
            
            # Filter by budget
            gaming_candidates = [laptop for laptop in gaming_candidates 
                                 if laptop.get('Price', float('inf')) <= budget]
            
            # Try to find high GPU laptops first
            high_gpu_laptops = [laptop for laptop in gaming_candidates 
                               if laptop.get('GPU', 'Medium').lower() == 'high']
            
            if high_gpu_laptops:
                gaming_candidates = high_gpu_laptops
            
            # If we still have too many, prioritize by gaming-related features
            if len(gaming_candidates) > 5:
                # Score laptops by gaming suitability
                scored_laptops = []
                for laptop in gaming_candidates:
                    score = 0
                    
                    # High GPU is best for gaming
                    if laptop.get('GPU', 'Medium').lower() == 'high':
                        score += 10
                    elif laptop.get('GPU', 'Medium').lower() == 'medium':
                        score += 5
                    
                    # Better processors get higher scores
                    if 'i9' in laptop.get('Processor', '').lower() or 'ryzen 9' in laptop.get('Processor', '').lower():
                        score += 8
                    elif 'i7' in laptop.get('Processor', '').lower() or 'ryzen 7' in laptop.get('Processor', '').lower():
                        score += 6
                    elif 'i5' in laptop.get('Processor', '').lower() or 'ryzen 5' in laptop.get('Processor', '').lower():
                        score += 4
                    
                    # More RAM is better for gaming
                    ram_size = int(laptop.get('RAM', '8GB').lower().replace('gb', '').strip())
                    score += min(ram_size // 4, 5)  # Up to 5 points for RAM
                    
                    # Gaming brands/models get bonus points
                    brand_model = f"{laptop.get('Brand', '')} {laptop.get('Model', '')}".lower()
                    gaming_keywords = ["rog", "predator", "legion", "omen", "g15", "g14", "gaming"]
                    if any(keyword in brand_model for keyword in gaming_keywords):
                        score += 5
                    
                    scored_laptops.append((laptop, score))
                
                # Sort by score (higher is better for gaming)
                scored_laptops.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 5
                gaming_candidates = [laptop for laptop, score in scored_laptops[:5]]
            
            if gaming_candidates:
                laptop_info = "\n".join([
                    f"- {laptop['Brand']} {laptop['Model']}: {laptop['Processor']}, {laptop['RAM']}, {laptop['Storage']}, {laptop['Display']}, ₹{laptop['Price']}"
                    for laptop in gaming_candidates
                ])
                
                response = f"Based on your requirements for a gaming laptop with high GPU performance, here are the best options within your budget:\n\n{laptop_info}\n\nWould you like more specific information about any of these models?"
                
                # Add user message to conversation history
                conversations[session_id].append({"role": "user", "content": user_input})
                # Add assistant response to conversation history
                conversations[session_id].append({"role": "assistant", "content": response})
                
                # Create conversation_bot for rendering
                conversation_bot = []
                for message in conversations[session_id][1:]:  # Skip system message
                    if message["role"] == "user":
                        conversation_bot.append({"user": message["content"]})
                    elif message["role"] == "assistant":
                        conversation_bot.append({"bot": message["content"]})
                    
                return render_template("conversation_bot.html", name_xyz=conversation_bot)
            
        # Handle open-ended requirement descriptions
        if "need" in lower_input or "looking for" in lower_input or "want" in lower_input or "require" in lower_input:
            # Check if this is already handled by other specific handlers
            if not (any(keyword in lower_input for keyword in inventory_keywords) or 
                    is_feature_query or is_comparison_query or 
                    "gaming" in lower_input and "gpu" in lower_input):
                
                # Extract user requirements using function calling
                user_requirements_string = get_user_requirement_string(user_input)
                
                # Check if we got a valid response
                if user_requirements_string and dictionary_present(user_requirements_string):
                    # Extract the dictionary from the string
                    try:
                        # Try to safely evaluate the string as a Python expression
                        user_requirements = ast.literal_eval(user_requirements_string)
                    except (SyntaxError, ValueError):
                        # If that fails, use regex to extract the dictionary
                        dict_match = re.search(r'\{[^\}]+\}', user_requirements_string)
                        if dict_match:
                            try:
                                user_requirements = ast.literal_eval(dict_match.group(0))
                            except:
                                user_requirements = {}
                        else:
                            user_requirements = {}
                    
                    # If we successfully extracted requirements
                    if user_requirements:
                        # Use the compare_laptops_with_user function to get recommendations
                        laptop_recommendations = compare_laptops_with_user(user_requirements)
                        
                        # Validate recommendations
                        validated_recommendations = recommendation_validation(laptop_recommendations)
                        
                        # Store recommendations for this session
                        recommendations[session_id] = validated_recommendations
                        
                        if validated_recommendations:
                            # Format recommendations for display
                            reco_text = "\n".join([
                                f"- {laptop['Brand']} {laptop['Model']}: {laptop['Processor']}, {laptop['RAM']}, {laptop['Storage']}, {laptop['Display']}, ₹{laptop['Price']}"
                                for laptop in validated_recommendations
                            ])
                            
                            # Create response with recommendations
                            response = f"Based on your requirements, here are my top recommendations:\n\n{reco_text}\n\nWould you like more information about any of these models?"
                        else:
                            response = "I couldn't find any laptops that match your specific requirements. Could you please adjust your criteria? For example, you might need to increase your budget or be more flexible with some specifications."
                        
                        # Add user message to conversation history
                        conversations[session_id].append({"role": "user", "content": user_input})
                        # Add assistant response to conversation history
                        conversations[session_id].append({"role": "assistant", "content": response})
                        
                        # Create conversation_bot for rendering
                        conversation_bot = []
                        for message in conversations[session_id][1:]:  # Skip system message
                            if message["role"] == "user":
                                conversation_bot.append({"user": message["content"]})
                            elif message["role"] == "assistant":
                                conversation_bot.append({"bot": message["content"]})
                                
                        return render_template("conversation_bot.html", name_xyz=conversation_bot)
        
        # Continue with the regular flow for other queries
        prompt = 'Remember your system message and that you are an intelligent laptop assistant. So, you only help with questions around laptop.'
        
        # Add user message to conversation history
        conversations[session_id].append({"role": "user", "content": user_input + prompt})
        
        # Create conversation_bot for rendering
        conversation_bot = []
        for message in conversations[session_id][1:]:  # Skip system message
            if message["role"] == "user":
                conversation_bot.append({"user": message["content"].replace(prompt, "")})  # Remove prompt from display
            elif message["role"] == "assistant":
                conversation_bot.append({"bot": message["content"]})

        # Get response from model
        response_assistant = get_chat_model_completions(conversations[session_id])
        
        # Re-enable moderation check for assistant response
        moderation = moderation_check(response_assistant)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        logger.info(f"Generated response: {response_assistant}")

        # Add assistant response to conversation history
        conversations[session_id].append({"role": "assistant", "content": response_assistant})
        conversation_bot.append({"bot": response_assistant})
        
        # Render the conversation
        return render_template("conversation_bot.html", name_xyz=conversation_bot)
    except NameError as e:
        logger.error(f"NameError in request processing: {str(e)}")
        return render_template("conversation_bot.html", 
                              name_xyz=[{"bot": f"There was an issue with the application configuration. Please try again or contact support if the issue persists."}])
    except KeyError as e:
        logger.error(f"KeyError in request processing: {str(e)}")
        return render_template("conversation_bot.html", 
                              name_xyz=[{"bot": f"There was an issue with your request. Some required information was missing. Please try again."}])
    except ValueError as e:
        logger.error(f"ValueError in request processing: {str(e)}")
        return render_template("conversation_bot.html", 
                              name_xyz=[{"bot": f"I couldn't process your request due to an invalid value. Please check your input and try again."}])
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        return render_template("conversation_bot.html", 
                              name_xyz=[{"bot": f"An unexpected error occurred. Our team has been notified. Please try again later."}])

@app.route("/debug_dataset")
def debug_dataset():
    # No need to import LAPTOP_DATASET here since we imported it at the top
    return render_template("debug.html", laptops=LAPTOP_DATASET)

@app.route("/reset_rate_limit")
def reset_rate_limit():
    # Create a new session ID to reset rate limiting
    session['session_id'] = str(uuid.uuid4())
    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host= "0.0.0.0", port=5001)
