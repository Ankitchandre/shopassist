# Remove the mock imports
# from mock_llm import (...)

# Keep the existing imports
import json
import requests
import re
from data_loader import load_laptop_data

# Load the laptop dataset
LAPTOP_DATASET = load_laptop_data()

# Define the shop_assist_custom_functions for reference
shop_assist_custom_functions = [
    {
        'name': 'extract_user_info',
        'description': 'Extract user information from the input',
        'parameters': {
            'type': 'object',
            'properties': {
                'GPU intensity': {
                    'type': 'string',
                    'description': 'GPU intensity of the user requested laptop'
                },
                'Display quality': {
                    'type': 'string',
                    'description': 'Display quality of the user requested laptop'
                },
                'Portability': {
                    'type': 'string',
                    'description': 'The portability of the user requested laptop'
                },
                'Multitasking': {
                    'type': 'string',
                    'description': 'The multitasking ability of the user requested laptop'
                },
                'Processing speed': {
                    'type': 'string',
                    'description': 'The processing speed of the user requested laptop'
                },
                'Budget': {
                    'type': 'integer',
                    'description': 'The budget of the user requested laptop'
                }
            }
        }
    }
]

# Keep the extract_user_info function
def extract_user_info(GPU_intensity, Display_quality, Portability, Multitasking, Processing_speed, Budget):
    """
    The local function that we have written to extract the laptop information for user
    """
    return {
        "GPU intensity": GPU_intensity,
        "Display quality": Display_quality,
        "Portability": Portability,
        "Multitasking": Multitasking,
        "Processing speed": Processing_speed,
        "Budget": Budget
    }

def initialize_conversation():
    """Initialize conversation with system message and welcome message"""
    return [
        {"role": "system", "content": "You are ShopAssist AI, a helpful assistant for finding the perfect laptop."},
        {"role": "assistant", "content": "Hello! I'm ShopAssist AI. How can I help you find the perfect laptop today?"}
    ]

def get_api_key():
    """Get API key from file"""
    try:
        with open("OpenAI_API_Key.txt", 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading API key: {e}")
        return None

def get_chat_model_completions(messages):
    """Get completions from OpenAI chat model using direct API call"""
    try:
        api_key = get_api_key()
        if not api_key:
            return "Error: API key not found"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return "I'm sorry, I encountered an error. Please try again."
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return "I'm sorry, I encountered an error. Please try again."

def moderation_check(user_input):
    """Simple local moderation check for inappropriate content"""
    inappropriate_words = [
        "porn", "xxx", "sex", "nude", "naked", "fuck", "shit", "damn", "bitch",
        "asshole", "cunt", "dick", "cock", "pussy", "kill", "murder", "suicide",
        "terrorist", "bomb", "explosive", "weapon", "gun", "racist", "nazi"
    ]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = user_input.lower()
    
    # Check if any inappropriate words are in the text
    for word in inappropriate_words:
        if word in text_lower:
            return "Flagged"
    
    return "Not Flagged"

def intent_confirmation_layer(response_assistant):
    """Check if assistant has captured user intent"""
    try:
        api_key = get_api_key()
        if not api_key:
            return "No"  # Default to No if API key is missing
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that determines if the user's requirements have been captured."},
            {"role": "user", "content": f"Based on this assistant response, have all the user's laptop requirements been captured? Answer only Yes or No.\n\nAssistant response: {response_assistant}"}
        ]
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return "No"  # Default to No on error
    except Exception as e:
        print(f"Error in intent confirmation: {e}")
        return "No"  # Default to No on error

def get_user_requirement_string(user_input):
    """
    Extract user requirements from their input using function calling.
    
    Args:
        user_input (str): The user's input text
        
    Returns:
        str: A string representation of the user requirements dictionary
    """
    try:
        api_key = get_api_key()
        if not api_key:
            return None
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare the messages for function calling
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts laptop requirements from user queries."},
            {"role": "user", "content": user_input}
        ]
        
        # Define the function for extracting user requirements
        functions = [
            {
                "name": "extract_user_info",
                "description": "Extract user information from the input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "GPU intensity": {
                            "type": "string",
                            "description": "GPU intensity of the user requested laptop (low/medium/high)"
                        },
                        "Display quality": {
                            "type": "string",
                            "description": "Display quality of the user requested laptop (low/medium/high)"
                        },
                        "Portability": {
                            "type": "string",
                            "description": "The portability of the user requested laptop (low/medium/high)"
                        },
                        "Multitasking": {
                            "type": "string",
                            "description": "The multitasking ability of the user requested laptop (low/medium/high)"
                        },
                        "Processing speed": {
                            "type": "string",
                            "description": "The processing speed of the user requested laptop (low/medium/high)"
                        },
                        "Budget": {
                            "type": "integer",
                            "description": "The budget of the user requested laptop in rupees"
                        }
                    },
                    "required": ["GPU intensity", "Display quality", "Portability", "Multitasking", "Processing speed", "Budget"]
                }
            }
        ]
        
        # Make the API call
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "functions": functions,
            "function_call": {"name": "extract_user_info"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the function call arguments
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_call = choice["message"]["function_call"]
                    if "arguments" in function_call:
                        return function_call["arguments"]
        
        return None
    except Exception as e:
        print(f"Error in get_user_requirement_string: {e}")
        return None

def get_chat_completions_func_calling(messages, functions):
    """
    Get chat completions with function calling capability.
    
    Args:
        messages (list): List of message dictionaries
        functions (list): List of function definitions
        
    Returns:
        dict: The response from the API
    """
    try:
        api_key = get_api_key()
        if not api_key:
            return None
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "functions": functions
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error in API call: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error in get_chat_completions_func_calling: {e}")
        return None

def compare_laptops_with_user(user_requirements):
    """
    Compare user requirements with laptops in the dataset and return the best matches
    
    Args:
        user_requirements: Dictionary containing user requirements
        
    Returns:
        List of laptops that best match the user requirements
    """
    # Extract user requirements
    gpu_intensity = user_requirements.get("GPU intensity", "medium").lower()
    display_quality = user_requirements.get("Display quality", "medium").lower()
    portability = user_requirements.get("Portability", "medium").lower()
    multitasking = user_requirements.get("Multitasking", "medium").lower()
    processing_speed = user_requirements.get("Processing speed", "medium").lower()
    budget = user_requirements.get("Budget", 80000)
    
    # Map text values to numerical scores
    intensity_map = {"low": 1, "medium": 2, "high": 3}
    
    # Calculate match score for each laptop
    scored_laptops = []
    for laptop in LAPTOP_DATASET:
        # Skip laptops that exceed budget
        if laptop["Price"] > budget:
            continue
        
        # Calculate match score based on requirements
        score = 0
        
        # GPU match
        laptop_gpu = laptop.get("GPU", "Medium").lower()
        if laptop_gpu in intensity_map:
            score += abs(intensity_map.get(gpu_intensity, 2) - intensity_map.get(laptop_gpu, 2))
        
        # Display quality match
        laptop_display = laptop.get("Display_Quality", "Medium").lower()
        if laptop_display in intensity_map:
            score += abs(intensity_map.get(display_quality, 2) - intensity_map.get(laptop_display, 2))
        
        # Portability match
        laptop_portability = laptop.get("Portability", "Medium").lower()
        if laptop_portability in intensity_map:
            score += abs(intensity_map.get(portability, 2) - intensity_map.get(laptop_portability, 2))
        
        # Multitasking match
        laptop_multitasking = laptop.get("Multitasking", "Medium").lower()
        if laptop_multitasking in intensity_map:
            score += abs(intensity_map.get(multitasking, 2) - intensity_map.get(laptop_multitasking, 2))
        
        # Processing speed match
        laptop_processing = laptop.get("Processing_Speed", "Medium").lower()
        if laptop_processing in intensity_map:
            score += abs(intensity_map.get(processing_speed, 2) - intensity_map.get(laptop_processing, 2))
        
        # Add laptop with its score
        scored_laptops.append((laptop, score))
    
    # Sort laptops by score (lower is better)
    scored_laptops.sort(key=lambda x: x[1])
    
    # Return top 3 matches (or fewer if not enough matches)
    return [laptop for laptop, score in scored_laptops[:3]]

def recommendation_validation(recommendations):
    """Validate laptop recommendations"""
    # If no recommendations, return empty list
    if not recommendations:
        return []
    
    # In a real implementation, this would apply additional validation rules
    # For now, just return the recommendations as is
    return recommendations

def initialize_conv_reco(recommendations):
    """Initialize conversation with recommendations"""
    system_message = "You are ShopAssist AI, a helpful assistant for finding the perfect laptop."
    if recommendations and len(recommendations) > 0:
        laptop_details = "\n".join([
            f"- {laptop['Brand']} {laptop['Model']}: {laptop['Processor']}, {laptop['RAM']}, {laptop['Storage']}, {laptop['Display']}, ₹{laptop['Price']}"
            for laptop in recommendations
        ])
        system_message += f"\n\nRecommended laptops:\n{laptop_details}"
    
    return [
        {"role": "system", "content": system_message}
    ]

def dictionary_present(text):
    """
    Check if the text contains a Python dictionary representation.
    This is used to verify if the chatbot has captured the user's profile as a structured dictionary.
    
    Args:
        text (str): The text to check for dictionary presence
        
    Returns:
        bool: True if a dictionary is present, False otherwise
    """
    # Look for patterns like {key: value} or {"key": value}
    import re
    
    # Pattern for dictionary-like structures
    dict_pattern = r'\{[^\}]+\}'
    
    # Check if the pattern exists in the text
    if re.search(dict_pattern, text):
        return True
    
    return False
