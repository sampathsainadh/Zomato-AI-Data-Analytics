import re
import json
import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI

# Hosted Llama model on Groq (check console.groq.com for current model names).
# The 70b model writes better SQL. If you hit rate limits, switch to "llama-3.1-8b-instant".
MODEL = "llama-3.3-70b-versatile"

FORBIDDEN_WORDS = ['drop', 'delete', 'truncate', 'alter', 'update', 'insert', 'create',
                   'replace', 'grant', 'revoke', 'merge', 'call', 'copy', 'put', 'remove']
FORBIDDEN_PATTERN = re.compile(r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b", re.IGNORECASE)

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method"
]

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"],
)

SCHEMA = """
Tables available (Snowflake). Use bare table names, no database or schema prefix.

FCT_ORDERS(order_id, order_date, customer_id, restaurant_id, city, cuisine,
           payment_method, order_status, is_delivered, sales_amount, discount,
           delivery_fee, gst, customer_rating, delivery_time_min)
DIM_RESTAURANT(restaurant_id, restaurant_name, city, cuisine, rating, cost_for_two)
DIM_CUSTOMER(customer_id, customer_name, age, age_segment, gender, city)
MART_DAILY_CITY_REVENUE(order_date, city, orders, cancel_rate, gmv, aov)
MART_RESTAURANT_PERFORMANCE(restaurant_id, restaurant_name, city, cuisine,
                            orders, revenue, avg_customer_rating, cancel_rate)
MART_DELIVERY_SLA(city, order_hour, delivered_orders, p50_delivery_min, late_rate)


Note: gmv means delivered revenue. Prefer the MART_ tables when they fit the question.
"""

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert. Write ONE SELECT query that answers the question.

Rules:
- SELECT queries only, never modify data.
- Use only the tables and columns listed in the schema below. Never invent columns.
- Use bare table names (FCT_ORDERS, not ZOMATO.MARTS.FCT_ORDERS).
- Add a LIMIT of 100 or less, unless the question asks for a single total.
- Reply as JSON in this exact format: {{"sql": "your query here"}}

{SCHEMA}
"""


@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        account=st.secrets["SNOWFLAKE_ACCOUNT"],
        user=st.secrets["SNOWFLAKE_USER"],
        password=st.secrets["SNOWFLAKE_PASSWORD"],
        warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
        database=st.secrets["SNOWFLAKE_DATABASE"],
        schema="MARTS",
        role=st.secrets.get("SNOWFLAKE_ROLE", "DBT_ROLE"),
    )


def clean_sql(sql):
    """Remove code fences, database prefixes and trailing semicolons from model output."""
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.replace("ZOMATO.MARTS.", "").replace("ZOMATO.", "")
    return sql.strip().rstrip(";").strip()


def generate_sql(question, retries=2):
    """Ask the model for SQL. Sometimes the JSON breaks, so retry."""
    last_error = None
    for _ in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
            )
            answer = response.choices[0].message.content
            sql = json.loads(answer)["sql"]
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("Model returned an empty SQL query")
            return clean_sql(sql)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            last_error = e
    raise last_error


def is_safe(sql):
    lowered = sql.strip().lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False

    # Only one statement allowed
    if ";" in lowered:
        return False

    # Whole-word match, so columns like "order_date" or "updated_at" aren't blocked by mistake
    if FORBIDDEN_PATTERN.search(lowered):
        return False

    return True


def run_query(sql):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        return cursor.execute(sql).fetch_pandas_all()
    finally:
        cursor.close()


st.title("Chat with your Zomato Data")
st.caption(f"Ask in English, {MODEL} (via Groq) writes the SQL, Snowflake runs it")

with st.sidebar:
    st.header("Example Questions")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f" - {q}")

question = st.text_input("Enter your question here",
                         placeholder="e.g. Top 10 restaurants by revenue in Bangalore")


if question:
    try:
        with st.spinner("Writing SQL..."):
            sql = generate_sql(question)
    except Exception as e:
        st.error(f"The model did not return usable SQL. Try rephrasing your question. ({e})")
        st.stop()

    st.code(sql, language="sql")

    if not is_safe(sql):
        st.error("The generated SQL is not safe to run. Please modify your question.")

    else:
        try:
            df = run_query(sql)
            st.success(f"{len(df)} rows returned")
            st.dataframe(df, hide_index=True)

            if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                st.bar_chart(df, x=df.columns[0], y=df.columns[1])

        except Exception as e:
            st.error(f"Error running query: {e}")
