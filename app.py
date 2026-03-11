import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
"inflation":[3,4,6,10,15,40,100,5,7,25],
"debt":[30,40,50,60,70,80,90,35,45,75],
"gdp_growth":[3,2,4,1,2,-1,-3,3,2,0],
"unemployment":[5,6,7,8,9,10,12,5,6,9],
"risk":[0,0,0,1,1,2,2,0,1,2]
}

df = pd.DataFrame(data)

X = df[["inflation","debt","gdp_growth","unemployment"]]
y = df["risk"]

model = RandomForestClassifier()
model.fit(X,y)

st.title("AI Country Risk Analyzer")

st.write("A machine learning platform that analyzes macroeconomic indicators to estimate the investment risk of a country.")

st.subheader("Economic Indicators Used")

st.write("• Inflation Rate")
st.write("• Public Debt (% of GDP)")
st.write("• GDP Growth")
st.write("• Unemployment Rate")

inflation = st.slider("Inflation (%)",0,150)
debt = st.slider("Debt (% GDP)",0,150)
gdp = st.slider("GDP Growth (%)",-10,10)
unemployment = st.slider("Unemployment (%)",0,20)

prediction = model.predict([[inflation,debt,gdp,unemployment]])

risk_levels = ["Low Risk","Medium Risk","High Risk"]

st.write("Predicted Country Risk:",risk_levels[prediction[0]])

st.subheader("Risk Interpretation")

st.write("Low Risk → Stable macroeconomic conditions")
st.write("Medium Risk → Moderate economic uncertainty")
st.write("High Risk → High macroeconomic instability")

st.subheader("Country Risk Ranking (Example)")

countries = pd.DataFrame({
"Country":["Chile","Peru","Brazil","Argentina"],
"inflation":[4,3,6,140],
"debt":[38,33,80,90],
"gdp_growth":[2.5,2.8,1,-1],
"unemployment":[8,6,9,7]
})

predictions = model.predict(countries[["inflation","debt","gdp_growth","unemployment"]])

countries["Risk"] = [risk_levels[p] for p in predictions]

st.dataframe(countries)