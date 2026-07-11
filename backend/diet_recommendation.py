import streamlit as st
import json
import os

# Try to import API client for backend mode
try:
    from frontend.api_client import (
        get_diet_quick_suggestions as api_quick_suggestions,
        get_diet_personalized_plan as api_personalized_plan,
        get_api_key, check_backend_health,
    )
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False

# Import direct Groq client as fallback
client = None
_Groq = None
try:
    from groq import Groq as _Groq
except ImportError:
    pass

def _init_groq():
    global client
    if client is None and _Groq is not None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            try:
                client = _Groq(api_key=api_key)
            except Exception:
                pass

def diet_tab():

    st.header("🥗 Diet Recommendation")

    if (
        "report_text" not in st.session_state
        or not st.session_state["report_text"].strip()
    ):
        st.warning("Please provide a lab report before generating diet recommendations.")
        return

    # -------- MODE SELECTION --------
    mode = st.radio(
        "Choose Diet Type",
        ["Quick Suggestions", "Personalized Meal Plan"]
    )

    text = st.session_state["lab_analysis"]
    # ================= QUICK MODE =================
    if mode == "Quick Suggestions":

        if st.button("🥗 Generate Quick Diet"):

            with st.spinner("Generating suggestions..."):

                prompt = f"""
You are a nutrition expert.

Based on this lab report, suggest a diet.

Rules:
- High sugar → diabetic diet
- Low hemoglobin → iron-rich foods
- High creatinine → kidney-friendly foods
- Keep it simple in 4 to 5 lines 
- Prefer Indian foods

Use bullet points.

Lab Report:
{text}
"""

                use_api = HAS_API_CLIENT and get_api_key() and check_backend_health()
                if use_api:
                    suggestions = api_quick_suggestions(text)
                    result = "\n".join(f"- {s}" for s in suggestions)
                else:
                    _init_groq()
                    if client is None:
                        st.error("No Groq client available. Configure backend or GROQ_API_KEY.")
                        return
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048
                    )
                    result = response.choices[0].message.content

                st.subheader("Diet Suggestions")
                st.markdown(result)

    # ================= PERSONALIZED MODE =================
    else:

        st.subheader("⚙️ Customize Your Diet")

        diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian"])
        goal = st.selectbox("Goal", ["General Health", "Weight Loss", "Muscle Gain"])
        meals = st.selectbox("Meals per day", [3, 4, 5])
        allergies = st.text_input("Any allergies? (optional)")

        if st.button("🍽️ Generate Meal Plan"):

            with st.spinner("Creating your personalized plan..."):

                prompt = f"""
You are a professional nutritionist.

Create a personalized daily meal plan.

User Details:
- Diet: {diet_type}
- Goal: {goal}
- Meals per day: {meals}
- Allergies: {allergies}

Medical Conditions (from lab report):
{text}

Guidelines:
- Adjust diet based on medical conditions
- Suggest Indian foods
- Keep meals practical and realistic
- Return BOTH a detailed text plan AND a JSON formatted macro breakdown per meal.

Return EXACTLY in this format:

JSON:
[
  {{"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15}},
  {{"meal": "Lunch", "calories": 600, "carbs": 70, "protein": 35, "fat": 20}},
  {{"meal": "Dinner", "calories": 500, "carbs": 40, "protein": 40, "fat": 15}},
  {{"meal": "Snacks", "calories": 200, "carbs": 25, "protein": 5, "fat": 10}}
]

PLAN:
Your detailed meal plan schedule and recommendations.
"""

                use_api = HAS_API_CLIENT and get_api_key() and check_backend_health()
                if use_api:
                    plan_data = api_personalized_plan(
                        report_text=text,
                        diet_type=diet_type,
                        goal=goal,
                        meals=meals,
                        allergies=allergies or "",
                    )
                    # Format as JSON:PLAN: for consistent parsing below
                    breakdown = plan_data.get("breakdown", [])
                    plan_text = plan_data.get("plan", "")
                    result = f"JSON:\n{json.dumps(breakdown, indent=2)}\n\nPLAN:\n{plan_text}"
                else:
                    _init_groq()
                    if client is None:
                        st.error("No Groq client available. Configure backend or GROQ_API_KEY.")
                        return
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048
                    )
                    result = response.choices[0].message.content
                st.session_state["personalized_diet_result"] = result

        if "personalized_diet_result" in st.session_state:
            result = st.session_state["personalized_diet_result"]
            try:
                import json
                import pandas as pd
                import plotly.express as px
                
                # Try to parse the JSON and Text
                json_part = result.split("JSON:")[1].split("PLAN:")[0].strip()
                plan_part = result.split("PLAN:")[1].strip()
                
                if "```json" in json_part:
                    json_part = json_part.split("```json")[1].split("```")[0].strip()
                elif "```" in json_part:
                    json_part = json_part.split("```")[1].split("```")[0].strip()
                
                macros_data = json.loads(json_part.strip())
                df_macros = pd.DataFrame(macros_data)
                
                st.subheader("📊 Meal-by-Meal Macronutrient Breakdown")
                fig = px.bar(
                    df_macros, 
                    x="meal", 
                    y=["carbs", "protein", "fat"],
                    title="Macronutrients (Grams) per Meal",
                    labels={"value": "Grams", "variable": "Macro"},
                    barmode="stack",
                    color_discrete_sequence=["#F4D03F", "#E74C3C", "#3498DB"]
                )
                st.plotly_chart(fig)
                
                st.subheader("🍽️ Your Personalized Meal Plan")
                st.markdown(plan_part)
            except Exception as e:
                # Fallback if AI response format isn't perfect
                st.subheader("🍽️ Your Personalized Meal Plan")
                st.markdown(result)