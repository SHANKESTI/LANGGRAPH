from pydantic import BaseModel,EmailStr,model_validator
from typing import List,Dict

class Patient(BaseModel):
    name:str
    email:str
    age:int
    weight:float
    married:bool
    allergies:List[str]
    contact_details:Dict[str,str]

    @model_validator(mode='after')
    def validate_emergency_contact(cls,model):
        if model.age>60 and 'emergency' not in model.contact_details:
            raise ValueError("Patient Older than 60 must have an emergency contact")
        return model
    
def update_patient(patient:Patient):
    print(patient.name)
    print(patient.email)
    print(patient.age)
    print(patient.weight)
    print(patient.married)
    print(patient.allergies)
    print(patient.contact_details)

patient_info={"name":"Shan","email":"Shankesti@gmail.com","age":64,"weight":53,"married":True,"allergies":['DUst',"NOSE"],"contact_details":{"number":"7795721008","emergency":"9242781008"}}
patient1=Patient(**patient_info)

print(update_patient(patient1))