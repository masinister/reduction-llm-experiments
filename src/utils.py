"""Utility functions for working with Pydantic models."""

from pydantic import BaseModel


def model_field_descriptions(model: type[BaseModel]) -> str:
    """Generate a bulleted list of field descriptions from a Pydantic model.
    
    Args:
        model: A Pydantic model class with Field descriptions.
        
    Returns:
        A string with bulleted descriptions for each field.
        
    Example:
        >>> class Person(BaseModel):
        ...     name: str = Field(description="The person's full name")
        ...     age: int = Field(description="Age in years")
        >>> print(model_field_descriptions(Person))
        - name: The person's full name
        - age: Age in years
    """
    lines = []
    for field_name, field_info in model.model_fields.items():
        description = field_info.description or "No description"
        lines.append(f"- {field_name}: {description}")
    return "\n".join(lines)
