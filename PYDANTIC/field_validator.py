from pydantic import BaseModel,EmailStr,AnyUrl,Field,field_validator
from typing import List,Dict,Optional,Annotated

class Patient(BaseModel):
    name:str
    email:EmailStr
    age:int
    weight:float
    married:bool
    allergies:List[str]
    contact_details:Dict[str,str]


    @field_validator('email')
    @classmethod
    def email_validator(cls,value):
        valid_domains=['hdfc.com','icici.com']
        domain_name=value.split('@')[-1]

        if domain_name not in valid_domains:
            raise ValueError("Not a valid domain")
        return value
    
    @field_validator('name')
    @classmethod
    def transform_name(cls,value):
        return value.upper()
    
    @field_validator('age',mode='after')
    @classmethod
    def validate_age(cls,value):
        if 0<value<100:
            return value
        raise ValueError('Age Should be in between 0 and 100')
    
    
def upadate_patient(patient:Patient):
    print(patient.name)
    print(patient.email)
    print(patient.age)
    print(patient.married)
    print("UPDATED")

patient_info={"name":"SHAN","email":"shankesti@hdfc.com","age":25,'weight':75.2,"married":True,"allergies":['dust','nose'],'contact_details':{"mobile_number":'7795721008'}}  

patient1=Patient(**patient_info)

print(upadate_patient(patient1))

