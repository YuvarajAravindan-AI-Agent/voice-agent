"""
Voice-agent tool functions calling the odoo-tools service (localhost:8790,
same VPS). Pipecat derives each tool's schema from the function's type hints
and docstring, so these stay plain async functions rather than manual
FunctionSchema definitions.
"""
import os

import aiohttp

from pipecat.services.llm_service import FunctionCallParams

ODOO_TOOLS_URL = os.getenv("ODOO_TOOLS_URL", "http://127.0.0.1:8790")
ODOO_TOOLS_TOKEN = os.getenv("ODOO_TOOLS_TOKEN")

_headers = {"Authorization": f"Bearer {ODOO_TOOLS_TOKEN}"}


async def get_products(params: FunctionCallParams, query: str = ""):
    """Look up products available for sale, optionally filtered by name.

    Args:
        query: Optional search term to filter products by name. Leave empty to list all.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{ODOO_TOOLS_URL}/products", params={"query": query}, headers=_headers
        ) as resp:
            data = await resp.json()
    await params.result_callback(data)


async def create_partner(params: FunctionCallParams, name: str, phone: str = "", email: str = ""):
    """Create a new customer record.

    Args:
        name: The customer's full name.
        phone: The customer's phone number, if given.
        email: The customer's email address, if given.
    """
    body = {"name": name}
    if phone:
        body["phone"] = phone
    if email:
        body["email"] = email
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{ODOO_TOOLS_URL}/partners", json=body, headers=_headers
        ) as resp:
            data = await resp.json()
    await params.result_callback(data)


async def create_order(
    params: FunctionCallParams, partner_id: int, product_id: int, quantity: float
):
    """Create a new sales order (quotation) for a customer with one product line.

    Args:
        partner_id: The id of the customer this order is for.
        product_id: The id of the product being ordered.
        quantity: How many units of the product to order.
    """
    body = {"partner_id": partner_id, "lines": [{"product_id": product_id, "quantity": quantity}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{ODOO_TOOLS_URL}/orders", json=body, headers=_headers) as resp:
            data = await resp.json()
    await params.result_callback(data)


async def get_order(params: FunctionCallParams, order_id: int):
    """Look up a single sales order by its id, including line items.

    Args:
        order_id: The id of the order to look up.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ODOO_TOOLS_URL}/orders/{order_id}", headers=_headers) as resp:
            data = await resp.json()
    await params.result_callback(data)


async def get_orders(params: FunctionCallParams, partner_id: int):
    """Look up all sales orders for a given customer.

    Args:
        partner_id: The id of the customer whose orders to look up.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{ODOO_TOOLS_URL}/orders", params={"partner_id": partner_id}, headers=_headers
        ) as resp:
            data = await resp.json()
    await params.result_callback(data)
