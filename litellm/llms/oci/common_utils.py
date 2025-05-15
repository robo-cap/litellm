import base64
import hashlib
import json
import os

import httpx

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

from litellm.litellm_core_utils.prompt_templates import factory as ptf
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.oci import OCIGenAIParams, OCIGenAIEndpoint



class OCIGenAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[Dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


def _get_oci_params(
    params: dict,
) -> OCIGenAIParams:
    """
    Get OCI GenAI parameters and return OCIGenAIParams object.
    """
    # Load auth variables from params
    auth_type = params.pop("auth_type", "API_KEY")
    auth_profile = params.pop("auth_profile", params.pop("oci_auth_profile", None))
    config_file = params.pop("config_file", params.pop("oci_config_file", None))
    region_name = params.pop("region_name", params.pop("region", None))
    compartment_id = params.pop("compartment_id", params.pop("oci_compartment_id", None))
    custom_endpoint = params.pop("custom_endpoint", params.pop("oci_custom_endpoint", None))

    if region_name is None:
        region_name = params.pop(
            "oci_region_name", params.pop("oci_region", None)
        )

    # Load auth variables from environment variables
    if region_name is None:
        region_name = (
            get_secret_str("OCI_REGION")
            or get_secret_str("REGION")
        )

    if auth_profile is None:
        auth_profile = (
            get_secret_str("OCI_AUTH_PROFILE")
            or "DEFAULT"
        )

    if config_file is None:
        config_file = (
            get_secret_str("OCI_CONFIG_FILE")
            or "~/.oci/config"
        )

    if auth_type == "API_KEY":
        config_file_path = Path(config_file).expanduser()
        if not config_file_path:
            raise OCIGenAIError(
                status_code=401,
                message=f"Error: {config_file} doesn't exist. Set the config_file path using the OCI_CONFIG_FILE environment variable or `config_file` parameter.",
            )
        try:
            import oci
            config = oci.config.from_file(config_file, auth_profile)
            if region_name is None:
                region_name = config["region"]
            
        except ModuleNotFoundError:
            raise OCIGenAIError(
                status_code=401,
                message="Error: OCI SDK not installed. Install the OCI SDK by running `pip install oci`.",
            )
        
    if compartment_id is None:
        compartment_id = (
            get_secret_str("OCI_COMPARTMENT_ID")
            or get_secret_str("COMPARTMENT_ID")
        )
    
    if not compartment_id:
        raise OCIGenAIError(
            status_code=401,
            message="Error: compartment_id is required. Pass in the compartment_id as a parameter or set it in the OCI_COMPARTMENT_ID environment variable.",
        )

    if region_name is None:
        raise OCIGenAIError(
            status_code=401,
            message="Error: region_name is required. Pass in the region_name as a parameter or set it in the OCI_REGION environment variable.",
        )
    
    return OCIGenAIParams(
        auth_type=auth_type,
        auth_profile=auth_profile,
        config_file=config_file,
        region_name=region_name,
        compartment_id=compartment_id,
        custom_endpoint=custom_endpoint
    )

# Mixin class for shared OCI GenAI functionality
class OCIBaseLLM:
    def __init__(self) -> None:
        self.body_signer = None
        # self.oci_params = _get_oci_params(kwargs.get("optional_params", {}))
        self.region_name = None
        # self._get_body_signer(self.oci_params)

    def _initialize_body_signer(
        self, oci_params: OCIGenAIParams,
    ) -> None:
        try:
            import oci
            if oci_params["auth_type"].upper() == "API_KEY":
                config = oci.config.from_file(
                    oci_params["config_file"],
                    oci_params["auth_profile"]
                )
                signer = oci.Signer.from_config(config)
                self.body_signer = signer._body_signer
            
            if oci_params["auth_type"].upper() == "INSTANCE_PRINCIPAL":
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                self.body_signer = signer._body_signer

            if oci_params["auth_type"].upper() == "RESOURCE_PRINCIPAL":
                os.environ["OCI_RESOURCE_PRINCIPAL_REGION"] = oci_params["region_name"]
                signer = oci.auth.signers.get_resource_principals_signer()
                self.body_signer = signer._body_signer
            
            if oci_params["auth_type"].upper() == "WORKLOAD_PRINCIPAL":
                os.environ["OCI_RESOURCE_PRINCIPAL_REGION"] = oci_params["region_name"]
                signer = oci.auth.signers.get_oke_workload_identity_resource_principal_signer()
                self.body_signer = signer._body_signer
            
        except ModuleNotFoundError:
            raise
        except Exception as e:
            raise OCIGenAIError(
                status_code=401,
                message=f"Error: Invalid authentication parameters. Message: {e}",
            )
        
    def _get_base_url(self, inference_mode: str) -> str:
        
        url = None
        if inference_mode.lower() == "chat":
            url = format(OCIGenAIEndpoint.DEFAULT_ENDPOINT, self.region_name) + OCIGenAIEndpoint.CHAT
        elif inference_mode.lower() == "embed_text":
            url = format(OCIGenAIEndpoint.DEFAULT_ENDPOINT, self.region_name) + OCIGenAIEndpoint.EMBED_TEXT
        elif inference_mode.lower() == "generate_text":
            url = format(OCIGenAIEndpoint.DEFAULT_ENDPOINT, self.region_name) + OCIGenAIEndpoint.GENERATE_TEXT
        elif inference_mode.lower() == "rerank_text":
            url = format(OCIGenAIEndpoint.DEFAULT_ENDPOINT, self.region_name) + OCIGenAIEndpoint.RERANK_TEXT
        elif inference_mode.lower() == "summarize_text":
            url = format(OCIGenAIEndpoint.DEFAULT_ENDPOINT, self.region_name) + OCIGenAIEndpoint.SUMMARIZE_TEXT
        
        if url is None:
            raise OCIGenAIError(
                status_code=401,
                message="Error: Could not generate url base. Invalid inference mode.",
            )
        return url

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return OCIGenAIError(
            status_code=status_code, message=error_message, headers=headers
        )

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        return headers
    
    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> dict:
        try:
            if self.body_signer is None:
                oci_params = _get_oci_params(optional_params)
                self._initialize_body_signer(oci_params)
        except Exception as e:       
            raise OCIGenAIError(
                status_code=401,
                message=f"Error: Body signer not set. Could not sign request {e}.",
            )
        # Injecting missing headers
        
        new_headers = { k.lower(): v for k, v in headers.items() }
        new_headers.setdefault("content-type", "application/json")
        new_headers.setdefault("date", datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"))
        new_headers.setdefault("host", urlparse(api_base).netloc)
        if "x-content-sha256" not in new_headers:
            try:
                body = json.dumps(request_data, allow_nan=False)
            except ValueError as e:
                raise OCIGenAIError(
                    status_code=400,
                    message=f"Error: Could not serialize request data. Message: {e}",
                )
            encoded_body = body.encode("utf-8")

            new_headers.setdefault("content-length", str(len(encoded_body)))
            
            m = hashlib.sha256()
            m.update(encoded_body)
            base64digest = base64.b64encode(m.digest())
            base64string = base64digest.decode("utf-8")
            new_headers["x-content-sha256"] = base64string
        
        if self.body_signer is None:
            raise OCIGenAIError(
                status_code=401,
                message="Error: Body signer is not initialized. Ensure that `_initialize_body_signer` is called before signing requests.",
            )
        
        signed_headers = self.body_signer.sign(
            headers=new_headers,
            method="POST",
            path=urlparse(api_base).path,
        )
        new_headers.update(signed_headers)
        return new_headers