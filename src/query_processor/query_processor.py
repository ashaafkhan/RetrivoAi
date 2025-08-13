import os
import json
import boto3
import logging
import psycopg2
import asyncio
import httpx
import time
import traceback
from typing import List, Dict, Any, Optional
from decimal import Decimal
from google import genai
from google.genai import types

# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
secretsmanager = boto3.client('secretsmanager')

# Environment variables
DOCUMENTS_BUCKET = os.environ.get('DOCUMENTS_BUCKET')
METADATA_TABLE = os.environ.get('METADATA_TABLE')
STAGE = os.environ.get('STAGE')
DB_SECRET_ARN = os.environ.get('DB_SECRET_ARN')
GEMINI_SECRET_ARN = os.environ.get('GEMINI_SECRET_ARN')
GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL')
TEMPERATURE = float(os.environ.get('TEMPERATURE'))
MAX_OUTPUT_TOKENS = int(os.environ.get('MAX_OUTPUT_TOKENS'))
TOP_K = int(os.environ.get('TOP_K'))
TOP_P = float(os.environ.get('TOP_P'))
ENABLE_EVALUATION = os.environ.get('ENABLE_EVALUATION', 'true').lower() == 'true'
GEMINI_MODEL = "gemini-2.0-flash"

# MCP Configuration
MCP_TIMEOUT = int(os.environ.get('MCP_TIMEOUT', '60'))
RAG_CONFIDENCE_THRESHOLD = float(os.environ.get('RAG_CONFIDENCE_THRESHOLD', '0.7'))
MIN_CONTEXT_LENGTH = int(os.environ.get('MIN_CONTEXT_LENGTH', '100'))

# Get Gemini API key from Secrets Manager
def get_gemini_api_key():
    try:
        response = secretsmanager.get_secret_value(SecretId=GEMINI_SECRET_ARN)
        return json.loads(response['SecretString'])['GEMINI_API_KEY']
    except Exception as e:
        logger.error(f"Error fetching Gemini API key: {str(e)}")
        raise

# Gemini client
try:
    GEMINI_API_KEY = get_gemini_api_key()
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Error configuring Gemini API client: {str(e)}")
    raise

# Custom JSON encoder for Decimal types to handle JSON serialization 
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

# Stateless MCP client for Lambda integration
class StatelessMCPClient:
    """
    Stateless MCP client for Lambda integration.
    Each request is independent and doesn't maintain session state.
    """
    
    def __init__(
        self,
        mcp_url: str,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream", # Accept both JSON and SSE responses
            **(headers or {})
        }
        self._request_id_counter = 0
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_id_counter += 1
        return f"req_{self._request_id_counter}_{int(time.time() * 1000)}"
    
    async def _make_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a stateless JSON-RPC request to the MCP server.
        
        Args:
            method: The MCP method to call
            params: Optional parameters for the method
            
        Returns:
            JSON-RPC response or error dict
        """
        request_id = self._generate_request_id()
        
        # Create JSON-RPC request
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                response = await http_client.post(
                    self.mcp_url,
                    json=jsonrpc_request,
                    headers=self.headers
                )
                response.raise_for_status()
                response_data = response.json()
                return {
                    "success": True,
                    "data": response_data,
                    "error": None
                }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "data": None,
                "error": f"HTTP error: {e.response.status_code}"
            }
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools from the MCP server."""
        return await self._make_request(method="tools/list")
    
    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        return await self._make_request(
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments or {}
            }
        )
    
    async def search_with_mcp(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform web search using MCP Server stateless client."""
        try:
            logger.info(f"Making stateless MCP request to: {self.mcp_url}")
            
            # Call the web_search tool
            result = await self.call_tool("web_search", {
                "query": query,
                "num_results": max_results
            })
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "data": None,
                    "source": "mcp_client"
                }
            
            # Extract search results from MCP response
            search_data = self._extract_tool_result(result["data"])
            
            logger.info("MCP search completed successfully")
            return {
                "success": True,
                "data": search_data,
                "source": "mcp_web_search"
            }
                    
        except Exception as e:
            logger.error(f"MCP search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "source": "mcp_client"
            }
    
    def _extract_tool_result(self, response_data) -> Any:
        """Extract tool result from MCP response."""
        try:
            if isinstance(response_data, dict):
                if "result" in response_data:
                    result = response_data["result"]
                    if isinstance(result, str):
                        return result
                    elif isinstance(result, dict):
                        if any(key in result for key in ['content', 'text', 'data', 'message']):
                            return str(result.get('content', result.get('text', result.get('data', result.get('message', str(result)))))) 
                        else:
                            return str(result)
                    else:
                        return str(result)
            elif "error" in response_data:
                error = response_data["error"]
                if isinstance(error, dict):
                    error_message = error.get('message', str(error))
                    error_code = error.get('code', 'unknown')
                    return f"MCP Error ({error_code}): {error_message}"
                else:
                    return f"MCP Error: {str(error)}"
        except Exception as e:
            logger.error(f"Error extracting tool result: {e}")
            return f"Error extracting result: {str(e)}"

# Lambda handler function
def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        body = {}
        if 'body' in event:
            if isinstance(event.get('body'), str) and event.get('body'):
                try:
                    body = json.loads(event['body'])
                except json.JSONDecodeError:
                    body = {}
            elif isinstance(event.get('body'), dict):
                body = event.get('body')
        
        # Extract parameters
        query = body.get('query')
        user_id = body.get('user_id', 'system')
        enable_evaluation = body.get('enable_evaluation', ENABLE_EVALUATION)
        model_name = body.get('model_name', GEMINI_MODEL)
        web_search_with_mcp = body.get('web_search_with_mcp', False)
        mcp_server_url = body.get('mcp_server_url', None)
        
        # Health check
        if event.get('action') == 'healthcheck' or body.get('action') == 'healthcheck':
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'message': 'Enhanced query processor with stateless agentic RAG is healthy',
                    'stage': STAGE,
                    'mcp_server_url': mcp_server_url
                })
            }

        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'message': 'Query is required'})
            }

        # Initialize stateless MCP client
        mcp_client = None
        if mcp_server_url:
            mcp_client = StatelessMCPClient(mcp_server_url, MCP_TIMEOUT)

        # Traditional RAG
        query_embedding = embed_query(query)
        relevant_chunks = similarity_search(query_embedding, user_id)
        
        # Assess RAG quality
        rag_assessment = assess_rag_quality(relevant_chunks, query)
        
        # MCP Web search if needed
        web_search_data = None
        mcp_web_search_used = False
        web_search_error = None
        
        if rag_assessment["needs_web_search"] and (mcp_client or web_search_with_mcp):
            logger.info(f"Triggering stateless agentic search. Reason: {rag_assessment['reason']}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                search_result = loop.run_until_complete(perform_mcp_web_search(query, mcp_client))
                if search_result["success"]:
                    web_search_data = search_result["data"]
                    mcp_web_search_used = True
                else:
                    web_search_error = search_result["error"]
            finally:
                loop.close()
        
        # Generate response
        response = generate_response(model_name, query, relevant_chunks, web_search_data)
        
        # Evaluation
        evaluation_results = {}
        if enable_evaluation:
            evaluation_contexts = relevant_chunks.copy()
            if web_search_data:
                evaluation_contexts.append({"content": str(web_search_data)})
            
            evaluation_results = evaluate_rag_response(
                model_name,
                query=query,
                answer=response,
                contexts=evaluation_contexts
            )

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'query': query,
                'response': response,
                'traditional_rag': {'results': relevant_chunks, 'assessment': rag_assessment},
                'mcp_web_search': {'used': mcp_web_search_used, 'data': web_search_data, 'error': web_search_error},
                'evaluation': evaluation_results
            }, cls=DecimalEncoder)
        }

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'message': f"Internal error: {str(e)}"})
        }
