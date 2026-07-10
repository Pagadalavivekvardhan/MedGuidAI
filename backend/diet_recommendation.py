import streamlit as st
import google.generativeai as genai

import os
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)

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
    model = genai.GenerativeModel("gemini-2.5-flash")

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

                result = model.generate_content(prompt).text

                st.subheader("🥗 Diet Suggestions")
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

                result = model.generate_content(prompt).text
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