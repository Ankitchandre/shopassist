# Create mock functions to replace the OpenAI calls
import json
import random
from data_loader import load_laptop_data

# Load the laptop dataset
LAPTOP_DATASET = load_laptop_data()

# Sample laptop data for recommendations
MOCK_LAPTOPS = [
    {
        "Brand": "HP", 
        "Model": "EliteBook 840 G8", 
        "Processor": "Intel Core i7-1165G7", 
        "RAM": "16GB", 
        "Storage": "512GB SSD",
        "Display": "14-inch FHD",
        "Price": 75000,
        "GPU": "High",
        "Display_Quality": "High",
        "Portability": "Medium",
        "Multitasking": "High",
        "Processing_Speed": "High"
    },
    {
        "Brand": "Dell", 
        "Model": "XPS 13", 
        "Processor": "Intel Core i7-1185G7", 
        "RAM": "16GB", 
        "Storage": "1TB SSD",
        "Display": "13.4-inch 4K",
        "Price": 90000,
        "GPU": "Medium",
        "Display_Quality": "High",
        "Portability": "High",
        "Multitasking": "High",
        "Processing_Speed": "High"
    },
    {
        "Brand": "Lenovo", 
        "Model": "ThinkPad X1 Carbon", 
        "Processor": "Intel Core i5-1135G7", 
        "RAM": "8GB", 
        "Storage": "256GB SSD",
        "Display": "14-inch FHD",
        "Price": 65000,
        "GPU": "Low",
        "Display_Quality": "Medium",
        "Portability": "High",
        "Multitasking": "Medium",
        "Processing_Speed": "Medium"
    }
]

def get_chat_model_completions(messages):
    """Mock implementation of chat completions"""
    # Get the last user message
    user_message = ""
    for msg in reversed(messages):  # Check messages in reverse order to get the most recent
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    
    # Check if this is a follow-up question about a specific laptop
    if any(brand.lower() in user_message.lower() for brand in ["hp", "dell", "lenovo"]):
        for laptop in MOCK_LAPTOPS:
            if laptop["Brand"].lower() in user_message.lower():
                return f"The {laptop['Brand']} {laptop['Model']} features a {laptop['Processor']} processor, {laptop['RAM']} RAM, and {laptop['Storage']} storage. It has a {laptop['Display']} display and costs ₹{laptop['Price']}."
    
    # Check for gaming laptop with budget
    if "gaming" in user_message.lower() and "budget" in user_message.lower():
        return f"Based on your requirements for a gaming laptop with high GPU performance and a budget of 80,000 rupees, I recommend the {MOCK_LAPTOPS[0]['Brand']} {MOCK_LAPTOPS[0]['Model']}. It features a {MOCK_LAPTOPS[0]['Processor']} processor, {MOCK_LAPTOPS[0]['RAM']} RAM, and {MOCK_LAPTOPS[0]['Storage']} storage with a {MOCK_LAPTOPS[0]['Display']} display. The price is ₹{MOCK_LAPTOPS[0]['Price']}, which fits your budget.\n\nWould you like to know more about this laptop or would you prefer to see other options?"
    
    # Check for specific requirements
    if "budget" in user_message.lower() and ("requirement" in user_message.lower() or "need" in user_message.lower()):
        return "I understand you need a laptop with high GPU intensity for gaming, high display quality, medium portability, high multitasking, and high processing speed with a budget of 80000."
    
    # Check for gaming-specific questions
    elif "gaming" in user_message.lower() or "game" in user_message.lower():
        return "For gaming, you'll need a laptop with high GPU intensity. Could you also tell me about your requirements for display quality, portability, multitasking, processing speed, and your budget?"
    
    # Check for office/work questions
    elif "office" in user_message.lower() or "work" in user_message.lower():
        return "For office work, you might need medium multitasking capabilities. Could you also tell me about your requirements for GPU intensity, display quality, portability, processing speed, and your budget?"
    
    # Check for comparison requests
    elif "compare" in user_message.lower() or "difference" in user_message.lower():
        return f"When comparing the {MOCK_LAPTOPS[0]['Brand']} {MOCK_LAPTOPS[0]['Model']} and {MOCK_LAPTOPS[1]['Brand']} {MOCK_LAPTOPS[1]['Model']}, the {MOCK_LAPTOPS[0]['Brand']} has {MOCK_LAPTOPS[0]['RAM']} RAM while the {MOCK_LAPTOPS[1]['Brand']} has {MOCK_LAPTOPS[1]['RAM']} RAM. The {MOCK_LAPTOPS[1]['Brand']} has a better display ({MOCK_LAPTOPS[1]['Display']}), but the {MOCK_LAPTOPS[0]['Brand']} is more affordable at ₹{MOCK_LAPTOPS[0]['Price']}."
    
    # Default greeting/introduction
    else:
        return "Hello! I'm ShopAssist AI. I can help you find the perfect laptop based on your requirements. Could you tell me what you're looking for in terms of GPU performance, display quality, portability, multitasking needs, processing speed, and your budget?"

def moderation_check(user_input):
    """Mock implementation of moderation check"""
    # Check for inappropriate content (very basic check)
    inappropriate_words = ["inappropriate", "offensive", "rude", "hate", "violent"]
    if any(word in user_input.lower() for word in inappropriate_words):
        return "Flagged"
    return "Not Flagged"

def intent_confirmation_layer(response_assistant):
    """Mock implementation of intent confirmation"""
    # If the response contains budget, gaming, or recommendation keywords, assume requirements are captured
    if any(keyword in response_assistant.lower() for keyword in ["budget", "gaming", "recommend", "based on your requirements"]):
        return "Yes"
    return "No"

def get_user_requirement_string(response_assistant):
    """Mock implementation of user requirement extraction"""
    return "I need a laptop with high GPU intensity, high display quality, medium portability, high multitasking, high processing speed, and a budget of 80000."

def get_chat_completions_func_calling(input, include_budget):
    """Mock implementation of function calling"""
    return {
        "GPU intensity": "high",
        "Display quality": "high",
        "Portability": "medium",
        "Multitasking": "high",
        "Processing speed": "high",
        "Budget": 80000
    }

def compare_laptops_with_user(user_requirements):
    """Mock implementation of laptop comparison"""
    # Just return the first two laptops as recommendations
    return MOCK_LAPTOPS[:2]

def recommendation_validation(recommendations):
    """Mock implementation of recommendation validation"""
    return recommendations

def initialize_conversation():
    """Mock implementation of conversation initialization"""
    return [
        {"role": "system", "content": "You are ShopAssist AI, a helpful assistant for finding the perfect laptop."},
        {"role": "assistant", "content": "Hello! I'm ShopAssist AI. How can I help you find the perfect laptop today?"}
    ]

def initialize_conv_reco(recommendations):
    """Mock implementation of recommendation conversation"""
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