
<<<<<<< Updated upstream
response = completion(
  model="gpt-3.5-turbo",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
=======
import os 

import litellm


response = litellm.completion(
    model="oci/meta.llama-3.1-70b-instruct",
    messages=[{ "content": "Who are you?","role": "user"}],
    api_base="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/20231130/actions/chat",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaqi3if6t4n24qyabx5pjzlw6xovcbgugcmatavjvapyq3jfb4diqq"
)
print(response)

response = litellm.completion(
    model="oci/meta.llama-3.1-70b-instruct",
    messages=[{ "content": "Who are you?","role": "user"}],
    api_base="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/20231130/actions/chat",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaqi3if6t4n24qyabx5pjzlw6xovcbgugcmatavjvapyq3jfb4diqq",
    stream=True
)

for part in response:
    print(part)
>>>>>>> Stashed changes
