"""
Translation from OpenAI's `/chat/completions` endpoint to IBM WatsonX's `/text/chat` endpoint.

Docs: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
"""

from typing import Any, List, Optional, AsyncIterator, Iterator, Union
import httpx

from litellm import token_counter
from litellm.secret_managers.main import get_secret_str
from litellm.llms.base_llm.base_model_iterator import FakeStreamResponseIterator

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from litellm.llms.base_llm.chat.transformation import BaseConfig
from ..common_utils import OCIBaseLLM
from litellm.types.llms.openai import AllMessageValues

from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from ..common_utils import OCIGenAIError

class OCIChatConfig(OCIBaseLLM, BaseConfig):
    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        OCIBaseLLM.__init__(self, **kwargs)

    @classmethod
    def get_config(cls):
        return super().get_config()
    
    def get_supported_openai_params(self, model: str) -> List:
        return [
            "temperature",  # equivalent to temperature
            "max_tokens",  # equivalent to maxTokens
            "top_p",  # equivalent to topP
            "frequency_penalty",  # equivalent to presencePenalty
            "stop",  # equivalent to stop
            "seed",  # equivalent to seed
            "stream",  # equivalent to isStream
            "tool_choice",  # equivalent to toolChoice
            "tools",  # equivalent to tools
            "top_logprobs", # equivalent to logProbs
            "n", # equivalent to numGenerations
            "presence_penalty" # equivalent to presencePenalty
        ]


    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        messages: Optional[List[AllMessageValues]] = None,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "max_tokens":
                optional_params["maxTokens"] = value
            if param == "top_p":
                optional_params["topP"] = value
            if param == "frequency_penalty":
                optional_params["presencePenalty"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "seed":
                optional_params["seed"] = value
            if param == "stream":
                optional_params["stream"] = value

            # if param == "tools":
            #     optional_params["tools"] = value
            # if param == "tool_choice":
            #     _tool_choice_value = self.map_tool_choice_values(
            #         model=model, tool_choice=value, drop_params=drop_params  # type: ignore
            #     )
            #     if _tool_choice_value is not None:
            #         optional_params["tool_choice"] = _tool_choice_value
            if param == "top_logprobs":
                optional_params["logProbs"] = value
            if param == "n":
                optional_params["numGenerations"] = value
            if param == "presence_penalty":
                optional_params["presencePenalty"] = value
        return optional_params
    

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: Optional[str],
        optional_params: Optional[dict],
        litellm_params: Optional[dict],
        stream: Optional[bool] = None,
    ) -> str:
        url = self._get_base_url(inference_mode="chat")
        return url

    def _completions_to_model(self, compartment_id: str | None, model: str, prompt: list, optional_params: dict) -> dict:
        params = {}
        if frequency_penalty := optional_params.get("frequency_penalty"):
            params["frequencyPenalty"] = frequency_penalty
        if max_tokens := optional_params.get("max_tokens"):
            params["maxTokens"] = max_tokens
        if presence_penalty := optional_params.get("presence_penalty"):
            params["presencePenalty"] = presence_penalty    
        if temperature := optional_params.get("temperature"):
            params["temperature"] = temperature
        if top_k := optional_params.get("top_k"):
            params["topK"] = top_k
        if top_p := optional_params.get("top_p"):
            params["topP"] = top_p
        if stream := optional_params.get("stream", False):
            params["isStream"] = stream
        chat_request = {
            "apiFormat": "GENERIC",
            "messages": prompt,
        }
        
        chat_request.update(params)
        return {
            "compartmentId": compartment_id,
            "servingMode": {
                "modelId": model,
                "servingType": "ON_DEMAND"
            },
            "chatRequest": chat_request
        }


    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict
    ) -> dict:
        transformed_messages = []
        for message in messages:
            transformed_messages.append(
                {
                    "role": message.get("role").upper(),
                    "content": [
                        {
                            "type": "TEXT",
                            "text": message.get("content")
                        }
                    ]
                }
            )
        compartment_id = optional_params.get("compartment_id")
        config = self.get_config()
        
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v
        
        data = self._completions_to_model(
            compartment_id=compartment_id, model=model, prompt=transformed_messages, 
            optional_params=optional_params
        )
        print(data)
        return data
    
    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except httpx.HTTPStatusError as e:
            raise OCIGenai(
                message=str(e),
                status_code=raw_response.status_code,
            )
        except Exception as e:
            raise OCIGenAIError(
                message=str(e),
                status_code=422,
            )
        try:
            chat_response = completion_response.get('chatResponse')
            choices_list = []
            for item in chat_response["choices"]:
                if len(item["message"]["content"][0]["text"]) > 0:
                    message_obj = Message(content=item["message"]["content"][0]["text"], role=item["message"]["role"].lower())
                else:
                    message_obj = Message(content=None)
                choice_obj = Choices(
                    finish_reason=item["finishReason"],
                    index=item["index"] + 1,  # check
                    message=message_obj,
                )
                choices_list.append(choice_obj)
            model_response.choices = choices_list  # type: ignore

        except Exception as e:
            raise OCIGenAIError(
                message=str(e),
                status_code=422,
            )

        # Calculate Usage
        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content"))
        )
        model_response.model = model
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return OCIModelResponseIterator(
            model_response=streaming_response,
            json_mode=json_mode,
        )


class OCIModelResponseIterator(FakeStreamResponseIterator):
    def __init__(
        self,
        model_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        json_mode: Optional[bool] = False,
    ):
        super().__init__(
            model_response=model_response,
            json_mode=json_mode,
        )

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            print("Chunk:", chunk)
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            text = (
                chunk.get("outputs", "")[0]
                .get("data", "")
                .get("text", "")
                .get("raw", "")
            )

            index: int = 0

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")
