from pydantic import BaseModel,EmailStr,computed_field
from typing import List,Dict

class Patient(BaseModel):
    name:str
    email:EmailStr
    age:int
    weight:float
    height:float
    allergies:List[str]
    contact_details:Dict[str,str]

    @computed_field
    @property
    def bmi(self)->float:
        bmi=round(self.weight/(self.height**2),2)
        return bmi
    
def update_patient(patient:Patient):
    print(patient.name)
    print(patient.email)
    print(patient.age)
    print(patient.weight)
    print("BMI",patient.bmi)

patient_info={"name":"SHAN","age":22,"email":"shankesti@gmail.com","weight":52.4,"height":1.74,"allergies":['Dust','Nose'],'contact_details':{"number":"7795721008"}}

patient1=Patient(**patient_info)

print(update_patient(patient1))