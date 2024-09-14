from typing import Optional
from pydantic import BaseModel, ValidationError, Field

from config import set_environment
from langchain.chains import create_extraction_chain_pydantic
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader


class Experience(BaseModel):
    # the title doesn't seem to help at all.
    start_date: Optional[str] = Field(description="When the job or study started.")
    end_date: Optional[str] = Field(description="When the job or study ended.")
    description: Optional[str] = Field(description="What the job or study entailed.")
    country: Optional[str] = Field(description="The country of the institution.")


class Study(Experience):
    degree: Optional[str] = Field(description="The degree obtained or expected.")
    institution: Optional[str] = Field(description="The university, college, or educational institution visited.")
    country: Optional[str] = Field(description="The country of the institution.")
    grade: Optional[str] = Field(description="The grade achieved or expected.")


class WorkExperience(Experience):
    company: str = Field(description="The company name of the work experience.")
    job_title: Optional[str] = Field(description="The job title.")


class Resume(BaseModel):
    first_name: Optional[str] = Field(description="The first name of the person.")
    last_name: Optional[str] = Field(description="The last name of the person.")
    #linkedin_url: Optional[str] = Field(description="The url of the linkedin profile of the person.")
    email_address: Optional[str] = Field(description="The email address of the person.")
    #nationality: Optional[str] = Field(description="The nationality of the person.")
    skill: Optional[str] = Field(description="A skill listed or mentioned in a description.")
    #study: Optional[Study] = Field(description="A study that the person completed or is in progress of completing.")
    #work_experience: Optional[WorkExperience] = Field(description="A work experience of the person.")
    hobby: Optional[str] = Field(description="A hobby or recreational activity of the person.")


set_environment()


def test_pydantic_model():
    # Sample Data
    sample_data = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    }
    # Validate resume
    try:
        resume = Resume(**sample_data)
        print("Validation successful:", resume)
    except ValidationError as e:
        print("Validation error:", e)


test_pydantic_model()

pdf_file_path = "/Users/O60774/ws/gen-ai/aravind/ch4/laverne-resume.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
chain = create_extraction_chain_pydantic(pydantic_schema=Resume, llm=chat_model, verbose=True)
resp = chain.run(docs)
print(resp[0].dict())
