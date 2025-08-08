from typing import Dict, Any
import requests
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class WeatherAgent:
    """Agent that provides weather information"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def get_weather(self, location: str) -> str:
        """Get weather information for a location (mock implementation)"""
        # Mock weather data - in real app, you'd use a weather API
        mock_weather = {
            "new york": "Sunny, 72°F",
            "london": "Cloudy, 15°C", 
            "tokyo": "Rainy, 18°C",
            "paris": "Partly cloudy, 20°C"
        }
        
        location_lower = location.lower()
        weather = mock_weather.get(location_lower, f"Weather data not available for {location}")
        return f"Weather in {location}: {weather}"
    
    def process_message(self, message: str) -> str:
        """Process weather-related messages"""
        response = self.llm.invoke([HumanMessage(content=f"""
You are a weather assistant. The user asked: {message}

If they're asking about weather, extract the location and use this format:
"I'll check the weather for [location]"

If it's not a weather question, respond with:
"I only handle weather requests. Please ask about weather in a specific location."
""")])
        
        # Extract location if it's a weather request
        content = response.content.lower()
        if "i'll check the weather for" in content:
            # Simple location extraction
            import re
            location_match = re.search(r"i'll check the weather for (.+)", content)
            if location_match:
                location = location_match.group(1).strip()
                return self.get_weather(location)
        
        return response.content


class CalculatorAgent:
    """Agent that performs mathematical calculations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Simple safe evaluation for basic math
            import ast
            import operator
            
            # Supported operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(node)
            
            result = eval_expr(ast.parse(expression, mode='eval').body)
            return f"Result: {result}"
        except:
            return f"Error: Could not calculate '{expression}'. Please use basic math operations (+, -, *, /, **)"
    
    def process_message(self, message: str) -> str:
        """Process calculation-related messages"""
        response = self.llm.invoke([HumanMessage(content=f"""
You are a calculator assistant. The user asked: {message}

If they're asking for a calculation, extract the mathematical expression and respond with:
"I'll calculate: [expression]"

If it's not a math question, respond with:
"I only handle mathematical calculations. Please ask me to calculate something."
""")])
        
        # Extract calculation if it's a math request
        content = response.content.lower()
        if "i'll calculate:" in content:
            import re
            calc_match = re.search(r"i'll calculate: (.+)", content)
            if calc_match:
                expression = calc_match.group(1).strip()
                return self.calculate(expression)
        
        return response.content