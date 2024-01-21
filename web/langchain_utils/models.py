from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Define your desired data structure.
class AnswerModel(BaseModel):
    answer: str = Field(description="answer to the questions")