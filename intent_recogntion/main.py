import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from src.predict_intent import predict_intent

app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    req_data = await request.json()
    user_input = req_data['queryResult']['queryText']
    intent_name = req_data['queryResult']['intent']['displayName']

    # Fallback Intent check
    if "Default Fallback Intent" in intent_name:
        prediction = predict_intent(user_input, model_type="bert")
        return JSONResponse({
            "fulfillmentText": f"I'm not sure, but I think your intent is: {prediction}"
        })

    # Regular intent response (optional: forward to something else)
    return JSONResponse({
        "fulfillmentText": f"You said: {user_input}"
    })

if __name__ == "__main__":
    uvicorn.run("intent_webhook_fastapi:app", host="0.0.0.0", port=8000, reload=True)
