from pydantic import BaseModel, ValidationError


class Person(BaseModel):
    age: int
    name: str
    is_student: bool


data = {
    "name": "John",
    "age": 25,
    "is_student": True
}

try:
    john = Person(**data)
    print(john.dict())
except ValidationError as e:
    print(e)
