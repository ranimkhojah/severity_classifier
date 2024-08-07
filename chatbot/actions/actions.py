from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import torch

from actions.model_loader import tokenizer, model, device

class ValidateSeverityClassificationForm(Action):
    def name(self) -> Text:
        return "validate_severity_classification_form"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        required_slots = ["vehicle", "vehicle_status", "speed", "environment", "road_condition"]
        events = []

        for slot_name in required_slots:
            if tracker.get_slot(slot_name) is None:
                dispatcher.utter_message(text=f"Please provide the {slot_name.replace('_', ' ')}.")
                return [SlotSet("requested_slot", slot_name)]

        for slot_name in required_slots:
            events.append(SlotSet(slot_name, tracker.get_slot(slot_name)))

        return events

class ActionSubmitSeverityForm(Action):
    def name(self) -> Text:
        return "action_submit_severity_form"

    def predict_severity(self, description: Text) -> Text:
        global tokenizer, model, device
        inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=1).item()
        severity_mapping = {0: 'S0', 1: 'S1', 2: 'S2', 3: 'S3'}
        return severity_mapping[prediction]

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        vehicle = tracker.get_slot('vehicle')
        vehicle_status = tracker.get_slot('vehicle_status')
        speed = tracker.get_slot('speed')
        environment = tracker.get_slot('environment')
        road_condition = tracker.get_slot('road_condition')
        danger = tracker.get_slot('danger')

        # Debug: Print slot values to ensure they are correctly retrieved
        print(f"Vehicle: {vehicle}, Vehicle Status: {vehicle_status}, Speed: {speed}, Environment: {environment}, Road Condition: {road_condition}, Danger: {danger}")

        description = (f"The {vehicle} is {vehicle_status} and at a speed of {speed} km/h in a {environment} environment. "
                       f"The road condition is {road_condition} and the danger is {danger}.")
        try:
            severity = self.predict_severity(description)
            summary = (f"The severity of the situation is {severity}.")
            dispatcher.utter_message(text=summary)
        except Exception as e:
            # Log the error for debugging purposes
            summary = (f"Thank you. Here's the information you provided: "
                       f"Vehicle: {vehicle}, Status: {vehicle_status}, "
                       f"Speed: {speed}, Environment: {environment}, "
                       f"Road Condition: {road_condition}, Danger: {danger}. ")
            dispatcher.utter_message(text=summary)

            print(f"Error during severity prediction: {str(e)}")
            dispatcher.utter_message(text=f"An error occurred while predicting the severity: {str(e)}")

        return [SlotSet("vehicle", vehicle), 
                SlotSet("vehicle_status", vehicle_status), 
                SlotSet("speed", speed), 
                SlotSet("environment", environment), 
                SlotSet("road_condition", road_condition),
                SlotSet("danger", danger)]
