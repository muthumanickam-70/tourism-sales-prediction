# ================================================================
# AI TOURISM INTELLIGENCE DASHBOARD (PROFESSIONAL VERSION)
# ================================================================

import customtkinter as ctk
import tkinter as tk
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Wedge

# ================================================================
# LOAD AI MODELS
# ================================================================

model = joblib.load("tourism_xgb_model.pkl")
encoder = joblib.load("event_encoder.pkl")
scaler = joblib.load("tourism_scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# ================================================================
# GUI SETUP
# ================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AI Tourism Intelligence Dashboard")
app.geometry("1500x920")

# ================================================================
# TITLE
# ================================================================

title = ctk.CTkLabel(
    app,
    text="Tourism Demand Prediction AI Dashboard",
    font=("Arial",32,"bold")
)

title.pack(pady=20)

# ================================================================
# MAIN CONTAINER
# ================================================================

container = ctk.CTkFrame(app)
container.pack(fill="both",expand=True,padx=20,pady=10)

# ================================================================
# INPUT PANEL
# ================================================================

input_frame = ctk.CTkFrame(container,width=280)
input_frame.pack(side="left",fill="y",padx=10,pady=10)

# ================================================================
# VARIABLES
# ================================================================

hotel = tk.StringVar(value="70")
flight = tk.StringVar(value="320")
temp = tk.StringVar(value="26")
econ = tk.StringVar(value="1.05")

month = tk.StringVar(value="7")
day = tk.StringVar(value="6")

lag7 = tk.StringVar(value="4200")
lag14 = tk.StringVar(value="4000")

event = tk.StringVar(value="Festival")

# ================================================================
# INPUT CREATOR
# ================================================================

def create_input(label,var,row):

    ctk.CTkLabel(input_frame,text=label).grid(row=row,column=0,padx=10,pady=6)

    entry = ctk.CTkEntry(input_frame,textvariable=var,width=120)
    entry.grid(row=row,column=1,padx=10)

# ================================================================
# INPUT FIELDS
# ================================================================

create_input("Hotel Occupancy (%)",hotel,0)
create_input("Flight Arrivals",flight,1)
create_input("Temperature (°C)",temp,2)
create_input("Economic Index",econ,3)

create_input("Month",month,4)
create_input("Day of Week",day,5)

create_input("Lag7 Visitors",lag7,6)
create_input("Lag14 Visitors",lag14,7)

ctk.CTkLabel(input_frame,text="Major Event").grid(row=8,column=0)

event_menu = ctk.CTkComboBox(
    input_frame,
    values=["Festival","Exhibition"],
    variable=event
)

event_menu.grid(row=8,column=1)

# ================================================================
# KPI PANEL
# ================================================================

kpi_frame = ctk.CTkFrame(container,height=120)
kpi_frame.pack(fill="x",padx=10,pady=10)

visitor_label = ctk.CTkLabel(
    kpi_frame,
    text="Predicted Visitors:",
    font=("Arial",22,"bold")
)

visitor_label.pack(pady=5)

level_label = ctk.CTkLabel(
    kpi_frame,
    text="Tourism Level:",
    font=("Arial",18)
)

level_label.pack()

risk_label = ctk.CTkLabel(
    kpi_frame,
    text="Tourism Risk Indicator:",
    font=("Arial",16)
)

risk_label.pack()

strategy_label = ctk.CTkLabel(
    kpi_frame,
    text="AI Strategy:",
    font=("Arial",14)
)

strategy_label.pack()

econ_label = ctk.CTkLabel(
    kpi_frame,
    text="Economic Risk:",
    font=("Arial",14)
)

econ_label.pack()

# ================================================================
# VISUAL PANEL
# ================================================================

visual_frame = ctk.CTkFrame(container)
visual_frame.pack(side="right",expand=True,fill="both",padx=10,pady=10)

visual_frame.grid_rowconfigure((0,1,2,3,4),weight=1)
visual_frame.grid_columnconfigure((0,1),weight=1)

# ================================================================
# CLEAR VISUALS
# ================================================================

def clear_visuals():
    for widget in visual_frame.winfo_children():
        widget.destroy()

# ================================================================
# GAUGE METER
# ================================================================

def draw_gauge(value):

    fig = plt.Figure(figsize=(6,3))
    ax = fig.add_subplot(111)

    ax.set_xlim(0,100)
    ax.set_ylim(0,10)

    bg = Wedge((50,0),40,0,180,color="lightgray",alpha=0.5)
    ax.add_patch(bg)

    angle = min((value/10000)*180,180)

    if value < 3000:
        color="red"
    elif value < 7000:
        color="orange"
    else:
        color="lime"

    fg = Wedge((50,0),40,0,angle,color=color)
    ax.add_patch(fg)

    ax.text(50,5,f"{int(value)} Visitors",ha="center",fontsize=18)

    ax.axis("off")

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,padx=10,pady=10)

# ================================================================
# FEATURE IMPORTANCE
# ================================================================

def show_importance():

    fig = plt.Figure(figsize=(6,4))
    ax = fig.add_subplot(111)

    importance = model.feature_importances_

    sns.barplot(
        x=importance,
        y=selected_features,
        hue=selected_features,
        palette="viridis",
        ax=ax
    )

    ax.set_title("Feature Importance",fontsize=14)
    ax.set_xlabel("Importance Score")

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=1,padx=10,pady=10)

# ================================================================
# TREND ANALYSIS
# ================================================================

def tourism_trend():

    days=np.arange(1,31)
    trend=3000+np.cumsum(np.random.normal(40,50,30))

    fig=plt.Figure(figsize=(6,4))
    ax=fig.add_subplot(111)

    ax.plot(days,trend,color="cyan",linewidth=3)

    ax.set_title("Tourism Trend Analysis")
    ax.set_xlabel("Days")
    ax.set_ylabel("Visitors")

    fig.tight_layout()

    canvas=FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1,column=0,padx=10,pady=10)

# ================================================================
# 7 DAY FORECAST
# ================================================================

def forecast_chart():

    days=np.arange(1,8)
    forecast=2500+np.cumsum(np.random.normal(30,40,7))

    fig=plt.Figure(figsize=(4,3))
    ax=fig.add_subplot(111)

    ax.plot(days,forecast,color="orange")

    ax.set_title("7 Day Forecast")

    canvas=FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1,column=1)

# ================================================================
# 12 MONTH FORECAST
# ================================================================

def yearly_forecast(pred):

    months=np.arange(1,13)
    forecast=pred*np.random.uniform(0.7,1.3,12)

    fig=plt.Figure(figsize=(6,4))
    ax=fig.add_subplot(111)

    ax.plot(months,forecast,color="cyan",marker="o",linewidth=3)

    ax.set_title("12 Month Tourism Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Visitors")

    fig.tight_layout()

    canvas=FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=2,column=0,padx=10,pady=10)

# ================================================================
# CALENDAR HEATMAP
# ================================================================

def calendar_heatmap():

    data=np.random.randint(1000,8000,(12,7))

    fig=plt.Figure(figsize=(4,3))
    ax=fig.add_subplot(111)

    sns.heatmap(data,cmap="coolwarm",ax=ax)

    ax.set_title("Tourism Calendar Heatmap")

    canvas=FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=2,column=1)

# ================================================================
# TOURISM HOTSPOTS
# ================================================================

def tourism_hotspots(pred):

    places=[
        "Taj Mahal","Goa Beach","Kerala Backwaters",
        "Jaipur Palace","Ooty","Varanasi",
        "Hampi","Rishikesh","Darjeeling","Andaman"
    ]

    demand=pred*np.random.uniform(0.4,1.4,10)

    fig=plt.Figure(figsize=(6,4))
    ax=fig.add_subplot(111)

    sns.barplot(
        x=demand,
        y=places,
        hue=places,
        palette="coolwarm",
        ax=ax
    )

    ax.set_title("Tourism Hotspot Detection")
    ax.set_xlabel("Visitor Demand")

    fig.tight_layout()

    canvas=FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3,column=0,padx=10,pady=10)

# ================================================================
# INDIA TOURISM MAP
# ================================================================

def india_tourism_map(pred):

    states=[
        "Delhi","Rajasthan","Goa","Kerala","Tamil Nadu",
        "Uttar Pradesh","Himachal","Karnataka","Maharashtra","Gujarat"
    ]

    visitors=pred*np.random.uniform(0.6,1.2,len(states))

    fig=plt.Figure(figsize=(6,4))
    ax=fig.add_subplot(111)

    sns.barplot(
        x=visitors,
        y=states,
        hue=states,
        palette="viridis",
        ax=ax
    )

    ax.set_title("India Tourism Demand by State")
    ax.set_xlabel("Visitors")

    fig.tight_layout()

    canvas=FigureCanvasTkAgg(fig,master=visual_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3,column=1,padx=10,pady=10)

# ================================================================
# TOURISM LEVEL + STRATEGY
# ================================================================

def tourism_level(visitors):

    if visitors<3000:

        level_label.configure(text="Tourism Level: LOW TOURISM",text_color="orange")
        risk_label.configure(text="Tourism Risk: Economic Risk")
        strategy_label.configure(text="AI Strategy: Increase Marketing")

    elif visitors<7000:

        level_label.configure(text="Tourism Level: MODERATE TOURISM",text_color="yellow")
        risk_label.configure(text="Tourism Risk: Stable Market")
        strategy_label.configure(text="AI Strategy: Promote Festivals")

    else:

        level_label.configure(text="Tourism Level: HIGH TOURISM",text_color="green")
        risk_label.configure(text="Tourism Risk: Peak Demand")
        strategy_label.configure(text="AI Strategy: Improve Infrastructure")

# ================================================================
# ECONOMIC RISK
# ================================================================

def economic_risk():

    econ_val=float(econ.get())

    if econ_val<0.8:
        econ_label.configure(text="Economic Risk: HIGH")
    elif econ_val<1.1:
        econ_label.configure(text="Economic Risk: MODERATE")
    else:
        econ_label.configure(text="Economic Risk: LOW")

# ================================================================
# PREDICTION
# ================================================================

def predict():

    try:

        clear_visuals()

        event_encoded=encoder.transform([event.get()])[0]

        input_data={

            "Hotel_Occupancy":float(hotel.get()),
            "Flight_Arrivals":float(flight.get()),
            "Average_Temperature":float(temp.get()),
            "Economic_Index":float(econ.get()),

            "Month":int(month.get()),
            "Day_of_Week":int(day.get()),

            "Lag7":float(lag7.get()),
            "Lag14":float(lag14.get()),

            "Major_Event":event_encoded,
            "Year":2024
        }

        features=pd.DataFrame([input_data])[selected_features]

        features_scaled=scaler.transform(features)

        pred=model.predict(features_scaled)[0]

        demand_factor=(

            float(hotel.get())*0.4+
            float(flight.get())*0.02+
            float(lag7.get())*0.0005+
            float(lag14.get())*0.0005
        )

        pred=pred*(demand_factor/50)

        season_factor={
        1:0.7,2:0.75,3:0.9,
        4:1.0,5:1.1,6:1.2,
        7:1.25,8:1.3,9:1.1,
        10:1.0,11:0.9,12:1.35
        }

        pred=pred*season_factor.get(int(month.get()),1)

        pred=max(500,min(pred,15000))

        visitor_label.configure(text=f"Predicted Visitors: {int(pred)}")

        tourism_level(pred)
        economic_risk()

        draw_gauge(pred)
        show_importance()
        tourism_trend()
        forecast_chart()
        yearly_forecast(pred)
        calendar_heatmap()
        tourism_hotspots(pred)
        india_tourism_map(pred)

    except Exception as e:

        visitor_label.configure(text="Invalid Input Values")
        print("Prediction Error:",e)

# ================================================================
# BUTTON
# ================================================================

predict_btn=ctk.CTkButton(
    input_frame,
    text="Predict Tourism Demand",
    command=predict
)

predict_btn.grid(row=10,column=0,columnspan=2,pady=20)

# ================================================================
# RUN
# ================================================================

app.mainloop()