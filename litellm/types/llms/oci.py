import json
from enum import Enum
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class OCIGenAIParams(TypedDict):
    auth_type: Literal["API_KEY", "INSTANCE_PRINCIPAL", "RESOURCE_PRINCIPAL", "WORKLOAD_PRINCIPAL"]
    auth_profile: str
    config_file: str
    region_name: str
    compartment_id: str
    custom_endpoint: Optional[str]


class OCIGenAIEndpoint(str, Enum):
    DEFAULT_ENDPOINT = "https://inference.generativeai.{}.oci.oraclecloud.com"
    CHAT = "/20231130/actions/chat"
    EMBED_TEXT = "/20231130/actions/embedText"
    GENERATE_TEXT = "/20231130/actions/generateText"
    RERANK_TEXT = "/20231130/actions/rerankText"
    SUMMARIZE_TEXT = "/20231130/actions/summarizeText"