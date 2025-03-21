# Complete mock of the OpenAI library
class MockOpenAI:
    def __init__(self):
        self.api_key = None
        self.chat = MockChat()
        self.models = MockModels()
        self.moderations = MockModerations()

class MockChat:
    def completions(self):
        return self
    
    def create(self, model, messages, functions=None, function_call=None):
        from mock_llm import get_chat_model_completions
        content = get_chat_model_completions(messages)
        
        if functions and function_call:
            from mock_llm import get_chat_completions_func_calling
            return MockFunctionResponse()
        
        return MockResponse(content)

class MockModels:
    def list(self):
        return MockModelList()

class MockModelList:
    def __init__(self):
        self.data = [MockModel("gpt-3.5-turbo"), MockModel("text-davinci-003")]

class MockModel:
    def __init__(self, id):
        self.id = id

class MockModerations:
    def create(self, input):
        return {"results": [{"flagged": False}]}

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    def __init__(self, content):
        self.content = content
        self.function_call = None

class MockFunctionResponse:
    def __init__(self):
        self.choices = [MockFunctionChoice()]

class MockFunctionChoice:
    def __init__(self):
        self.message = MockFunctionMessage()

class MockFunctionMessage:
    def __init__(self):
        self.content = None
        self.function_call = MockFunctionCall()

class MockFunctionCall:
    def __init__(self):
        self.arguments = '{"GPU intensity": "medium", "Display quality": "high", "Portability": "medium", "Multitasking": "high", "Processing speed": "high", "Budget": 80000}'
        self.name = "extract_user_info"

# Create a ChatCompletion class for the old API style
class ChatCompletion:
    @staticmethod
    def create(model, messages, functions=None, function_call=None):
        from mock_llm import get_chat_model_completions
        content = get_chat_model_completions(messages)
        
        if functions and function_call:
            return MockFunctionResponse()
        
        return MockResponse(content)

# Create a Moderation class for the old API style
class Moderation:
    @staticmethod
    def create(input):
        return {"results": [{"flagged": False}]}

# Replace the entire openai module
import sys
sys.modules["openai"] = MockOpenAI() 