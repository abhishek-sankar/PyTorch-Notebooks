import yaml
import os
from typing import Dict, Any, List
from pathlib import Path

class PromptManager:
    """Manages YAML-based prompt templates for the migration system."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all YAML prompt templates from the prompts directory."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                
                template_name = yaml_file.stem
                self.templates[template_name] = template_data
                print(f"Loaded template: {template_name}")
                
            except Exception as e:
                print(f"Error loading template {yaml_file}: {e}")
    
    def get_prompt(self, template_name: str, variables: Dict[str, Any] = None) -> str:
        """
        Generate a prompt from a template with variable substitution.
        
        Args:
            template_name: Name of the template (without .yaml extension)
            variables: Dictionary of variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.templates:
            available = ', '.join(self.templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        template = self.templates[template_name]
        prompt_template = template['template']
        
        if variables is None:
            variables = {}
        
        # Validate required variables
        required_vars = [var['name'] for var in template.get('variables', []) if var.get('required', False)]
        missing_vars = [var for var in required_vars if var not in variables]
        
        if missing_vars:
            raise ValueError(f"Missing required variables for template '{template_name}': {missing_vars}")
        
        # Substitute variables in the template
        try:
            formatted_prompt = prompt_template.format(**variables)
            return formatted_prompt
        except KeyError as e:
            raise ValueError(f"Variable {e} not provided but used in template")
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get metadata about a template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        return {
            'name': self.templates[template_name]['name'],
            'description': self.templates[template_name]['description'],
            'variables': self.templates[template_name].get('variables', [])
        }
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def validate_variables(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate variables against template requirements.
        
        Returns:
            Dictionary with 'valid', 'missing_required', 'extra_variables' keys
        """
        if template_name not in self.templates:
            return {'valid': False, 'error': f"Template '{template_name}' not found"}
        
        template = self.templates[template_name]
        template_vars = {var['name']: var for var in template.get('variables', [])}
        
        required_vars = [name for name, var in template_vars.items() if var.get('required', False)]
        provided_vars = set(variables.keys())
        expected_vars = set(template_vars.keys())
        
        missing_required = [var for var in required_vars if var not in provided_vars]
        extra_variables = provided_vars - expected_vars
        
        return {
            'valid': len(missing_required) == 0,
            'missing_required': missing_required,
            'extra_variables': list(extra_variables),
            'template_variables': list(expected_vars)
        }

# Example usage and testing
if __name__ == "__main__":
    try:
        pm = PromptManager()
        
        print("Available templates:")
        for template in pm.list_templates():
            info = pm.get_template_info(template)
            print(f"  {template}: {info['description']}")
        
        # Test analysis template
        print("\n" + "="*50)
        print("Testing analysis template:")
        
        test_variables = {
            'code': 'public class Test { @Deprecated public void oldMethod() {} }',
            'java_version': '21',
            'dependencies': 'JUnit 5, Spring Boot 3.0'
        }
        
        prompt = pm.get_prompt('analysis', test_variables)
        print("Analysis prompt generated successfully!")
        print(f"Prompt length: {len(prompt)} characters")
        
    except Exception as e:
        print(f"Error: {e}")