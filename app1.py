
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
import datetime




# OrderStatusManager class
class OrderStatusManager:
    def __init__(self):
        self.order_statuses = {
            "12345": "Shipped",
            "67890": "Processing",
            "11223": "Delivered",
        }
        self.order_price = {
            "12345": "500$",
            "67890": "79$",
            "11223": "900$",
        }
        self.order_shipped = {
            "12345": "22/2/25",
            "67890": "24/2/25",
            "11223": "2/2/25",
        }
        self.original_prices = self.order_price.copy()

    def get_total_price(self, order_id: str) -> str:
        
        """Get the total price of an order.

        Args:
            order_id: The ID of the order.

        Returns:
            The total price of the order.
        """
        return self.order_price.get(order_id, "total sum price")

        

    def get_order_status(self, order_id: str) -> str:
        """Get the status of an order.

        Args:
            order_id: The ID of the order.

        Returns:
            The status of the order.
        """
        return self.order_statuses.get(order_id, "Order ID not found.")

    def initiate_return(self, order_id: str, reason: str) -> str:
        """Initiate a return for an order.

        Args:
            order_id: The ID of the order.
            reason: The reason for the return.

        Returns:
            A message confirming the return initiation.
        """
        if order_id in ["12345", "67890", "11223"]:
            shipping_date_str = self.order_shipped.get(order_id)
            if shipping_date_str:
                shipping_date = datetime.datetime.strptime(
                    shipping_date_str, "%d/%m/%y"
                ).date()
                days_since_shipped = (datetime.date.today() - shipping_date).days
                if days_since_shipped > 10:
                    return (
                        f"Return cannot be initiated for order {order_id} as it has been"
                        f" more than 10 days since shipping."
                    )
            self.order_statuses[order_id] = "Return Initiated"
            original_price = float(self.order_price[order_id].rstrip("$"))
            penalty = original_price * 0.02
            new_price = original_price - penalty
            self.order_price[order_id] = f"{new_price:.2f}$"
            return (
                f"Return initiated for order {order_id} due to: {reason}. Price after"
                f" penalty is {self.order_price[order_id]}."
            )
        else:
            return "Order ID not found. Cannot initiate return."

    def cancel_return(self, order_id: str, reason: str) -> str:
        """Cancel a return for an order.

        Args:
            order_id: The ID of the order.
            reason: The reason for canceling the return.

        Returns:
            A message confirming the return cancellation.
        """
        if order_id in self.order_statuses:
            if self.order_statuses[order_id] == "Return Initiated":
                self.order_statuses[order_id] = self.order_statuses.get(
                    order_id, "Cancelled Order"
                )
                self.order_price[order_id] = self.order_price.get(
                    order_id, "order_price"
                )
                return (
                    f"Return cancellation initiated for order {order_id} due to:"
                    f" {reason}. Price reset to {self.order_price[order_id]}."
                )
            else:
                return (
                    f"Order {order_id} is not in 'Return Initiated' status. Cannot"
                    " cancel return."
                )
        else:
            return "Order ID not found. Cannot cancel return."


# Instance of OrderStatusManager and tools
order_manager = OrderStatusManager()
get_total_price_tool = tool(order_manager.get_total_price)
get_order_status_tool = tool(order_manager.get_order_status)
initiate_return_tool = tool(order_manager.initiate_return)
cancel_return_tool = tool(order_manager.cancel_return)

# LLM and workflow setup
llm = ChatGroq(model="Llama3-8b-8192")
tools = [
    get_order_status_tool,
    initiate_return_tool,
    get_total_price_tool,
    cancel_return_tool,
]
llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Streamlit app
st.title("Order Status Chatbot")

# Tabs
tab1, tab_examples = st.tabs(["Chat", "Examples"])

# Chat Tab
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your message"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": "1"}}
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )

        for event in events:
            if "messages" in event:
                response_content = event["messages"][-1].content
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_content}
                )
                with st.chat_message("assistant"):
                    st.markdown(response_content)

# Examples Tab
with tab_examples:
    example = {
        "Orderid": "12345",
        "Questions": [
            "What is the current status of my order?",
            "I want to return this order because it's damaged.",
            "What is the total price of this order?",
            "I want to cancel the return for this order.",
        ],
    }
    st.write("**Example Order and Questions:**")
    st.write(example)
    
    if st.button("Use Example Order ID"):
        user_input = f"What is the status of order {example['Orderid']}?"
        # ... (rest of the code to process user_input remains the same) ...
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": "1"}}
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )

        for event in events:
            if "messages" in event:
                response_content = event["messages"][-1].content
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_content}
                )
                with st.chat_message("assistant"):
                    st.markdown(response_content)
