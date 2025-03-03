from google import genai

client = genai.Client(api_key="AIzaSyBTTng5YdiZ8Md3mniKLr4EdxwAoPD8gz4")
# response = client.models.generate_content_stream(
#     model="gemini-2.0-flash",
#     contents=["Explain how AI works"])
# for chunk in response:
#     print(chunk.text, end="")

while True:
    inputSent=input()
    try:
        if inputSent.lower()=='quit':
            print('time to quit')
        else:
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=[inputSent])
            for chunk in response:
                print(chunk.text, end="")
    except Exception as e:
        print('e')
