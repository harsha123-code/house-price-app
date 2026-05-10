import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sklearn.compose._column_transformer as ct
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import time
import joblib
import plotly.graph_objects as go
from duckduckgo_search import DDGS
import requests
import streamlit as st

class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList
st.set_page_config(
page_title="House Price Predictor",
layout="wide"
)






@st.cache_data(show_spinner=False)
def get_image_ddg(query):
    try:
        time.sleep(1)  # prevent rate limit

        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=1))

            if results:
                return results[0].get("image")
            else:
                return None

    except Exception as e:
        print("Image fetch error:", e)
        return None


st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #ffffff;
    }

    label {
        color: #000000 !important;
        font-weight: 500;
    }

    h1, h2, h3 {
        color: #000000;
    }

    div[data-baseweb="select"] {
        background-color: #ffffff;
    }

</style>
""", unsafe_allow_html=True)



model = joblib.load(
'xgb_pipeline.joblib'
)
    
df = pd.read_csv("data_before_model_selection.csv")
geo_df=pd.read_csv("geo_mapdataset.csv")

if "page" not in st.session_state:
    st.session_state.page = "Home"


st.sidebar.title(" Navigation")

selected = st.sidebar.radio(
    "Go to",
    ["Home", "Price Predictor", "Insight module", "Analysis", "Recommendation"],
    index=["Home", "Price Predictor", "Insight module", "Analysis", "Recommendation"].index(st.session_state.page)
)


st.session_state.page = selected

page = st.session_state.page

if page=="Home":
    
    st.title(" Mumbai Flat Price Predictor")
    st.caption("Predict prices, explore insights, and make smarter real estate decisions in Mumbai.")

    st.info(" ML-powered tool for price prediction, what-if analysis, and market insights.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(" Accurate Predictions", "ML Model")
    c2.metric(" What-if Analysis", "Insight Module")
    c3.metric(" flat recomendation", "Trends & Patterns")
    c4.metric(" Easy to Use", "Fast UI")

    st.markdown("###  What can you do here?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("####  Price Prediction")
        st.write("Estimate flat price based on key features.")
        if st.button("Predict Price"):
            st.session_state.page = "Price Predictor"

    with col2:
        st.markdown("####  Insight Module")
        st.write("See how changing one feature affects price.")
        if st.button("Explore Insights"):
            st.session_state.page = "Insight module"

    with col3:
        st.markdown("####  Data analysis")
        st.write("Analyze trends and patterns in Mumbai market.")
        if st.button("View analysis"):
            st.session_state.page = "Analysis"


    with col4:
        st.markdown("####  Recommendation")
        st.write("Get personalized flat recommendations.")
        if st.button("Get Recommendations"):
            st.session_state.page = "Recommendation"


    st.markdown("###  Mumbai Real Estate at a Glance")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(" Flats", "6K+")
    s2.metric(" Avg Price", "₹2.96 Cr")
    s3.metric(" Price Range", "₹1,440–94,120/sq.ft")
    s4.metric(" Localities", "50+")


    st.markdown("###  Why this matters")

    st.write("""
    - Location is the biggest factor affecting price  
    - Area (buildup + carpet) directly impacts cost  
    - Bedrooms & bathrooms influence demand  
    - Floor and furnishing add premium value  
    """)

    st.success(" Model Accuracy: ~95% R² Score")
    st.info(" This is an AI/ML prediction. Real prices may vary.")


            


elif page == "Price Predictor":

    st.title(" Mumbai Flat Price Predictor")

    # ---- LOAD MODEL ----
    model = joblib.load("xgb_pipeline.joblib")

    st.markdown("### 🔹 Enter Flat Details")

    # ---- ROW 1 ----
    col1, col2, col3 = st.columns(3)

    with col1:
        locality = st.selectbox(
            "Locality",
            sorted(df["locality"].dropna().unique())
        )
        locality = str(locality).strip()

    with col2:
        bedrooms = int(st.selectbox("Bedrooms", sorted(df["bedrooms"].unique())))

    with col3:
        bathrooms = int(st.selectbox("Bathrooms", sorted(df["bathrooms"].unique())))

    # ---- ROW 2 ----
    col4, col5, col6 = st.columns(3)

    with col4:
        furnish = st.selectbox(
            "Furnishing",
            ["Unfurnished", "Semi Furnished", "Fully Furnished"]
        )

    with col5:
        floor = st.selectbox(
            "Floor Category",
            ["Lower", "Middle", "Upper"]
        )

    with col6:
        age = st.selectbox(
            "Age Category",
            ["Very Old", "Old", "Moderate", "New"],
            index=2
        )

   
    col7, col8 = st.columns(2)

    with col7:
        buildup = float(st.slider(
            "Build-up Area",
            int(df["buildup_area"].min()),
            int(df["buildup_area"].max()),
            800
        ))

    with col8:
        st.write("")  

    
    carpet = buildup * 0.8
    total_floors = 10


    input_df = pd.DataFrame([{
        "buildup_area": buildup,
        "carpet_area": carpet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "furnish_detail": furnish,
        "age_category": age,
        "floor_category": floor,
        "total_floors": total_floors,
        "locality": locality
    }])

    
    if st.button("Predict Price"):

        price = model.predict(input_df)[0]

        st.markdown("### 📊 Prediction Result")

        price_full = price * 10000000

        st.markdown(f"""
        <div style="
            background-color:#e8f5e9;
            padding:20px;
            border-radius:12px;
            border:1px solid #c8e6c9;
        ">
            <h2 style="color:#2e7d32;">₹ {price:.2f} Cr</h2>
            <p style="color:gray;">(₹ {price_full:,.0f})</p>
            <b style="color:#388e3c;"> Good Choice!</b>
        </div>
        """, unsafe_allow_html=True)

        
        

elif page == "Analysis":

    st.title("📊 Data Analysis")

    # Step 1: Clean + prepare data
    geo_df.columns = geo_df.columns.str.strip().str.lower()
    cols = ['price', 'price_sqft', 'buildup_area', 'carpet_area', 'latitude', 'longitude']

    for col in cols:
        geo_df[col] = pd.to_numeric(geo_df[col], errors='coerce')

    geo_df = geo_df.dropna(subset=cols)

    group_df = geo_df.groupby('locality')[cols].mean().reset_index()

    # Step 2: Bubble size
    group_df['bubble_size'] = (
        (group_df['price'] - group_df['price'].min()) /
        (group_df['price'].max() - group_df['price'].min())
    ) * 20 + 5

    # Step 3: Create map
    fig = px.scatter_mapbox(
        group_df,
        lat='latitude',
        lon='longitude',
        size='bubble_size',
        color='price_sqft',
        hover_name='locality',
        text='locality',
        hover_data={
            'buildup_area': ':.0f',
            'carpet_area': ':.0f',
            'price_sqft': ':.0f',
            'latitude': ':.2f',
            'longitude': ':.2f',
            'bubble_size': False
        },
        custom_data=['buildup_area','carpet_area','price_sqft'],
        zoom=10,
        height=500,
        color_continuous_scale="Turbo"
    )

    # Step 4: Styling
    fig.update_traces(
        marker=dict(opacity=0.75),
        textposition="top center",
        hovertemplate=
        "<b>%{hovertext}</b><br>" +
        "Built-up: %{customdata[0]:.0f} sqft<br>" +
        "Carpet: %{customdata[1]:.0f} sqft<br>" +
        "Price/sqft: %{customdata[2]:.0f}<br>" +
        "<extra></extra>"
    )
    fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":20,"l":0,"b":0},
)

    fig.update_geos(fitbounds="locations")

    society_df=pd.read_csv(r"C:\Users\DELL\Desktop\house price predicton\flats_final")
    society_df['society_name'] = society_df['society_name'].str.lower().str.strip()
    group_society = society_df.groupby('society_name').agg({
        'price': 'median',
        'society_name': 'count'
    }).rename(columns={'society_name': 'count'})
    group_society = group_society[group_society['count'] >= 5]
    
    group_society = group_society[~group_society.index.str.contains(
        'project|flat|bhk|apartment', case=False, na=False
    )]

    freq = group_society['price'].to_dict()

    wc = WordCloud(
        width=600,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(freq)

    fig_wc, ax = plt.subplots(figsize=(4,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')



    st.subheader("🌍 Price Distribution Map")
    st.plotly_chart(fig, use_container_width=True, height=1000)

    st.markdown("---")  

    
    st.subheader("🏢 Society Price Word Cloud")
    fig_wc, ax = plt.subplots(figsize=(4,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig_wc)
    df2 = pd.read_csv("flats_final")
    fig_scatter = px.scatter(
    df2,
    x='buildup_area',
    y='price',
    color='price_sqft',
    color_continuous_scale='plasma',

    hover_data=['locality'],  

    labels={
        'buildup_area': 'Area (sqft)',
        'price': 'Price (Cr)',
        'price_sqft': 'Price per sqft'
    },

    
)

    fig_scatter.update_traces(
        customdata=df2[['locality']],   #  THIS WAS MISSING
        hovertemplate=
        "<b>%{customdata[0]}</b><br>" +
        "Area: %{x} sqft<br>" +
        "Price: %{y:.2f} Cr<br>" +
        "Price/sqft: %{marker.color:.0f}<br>" +
        "<extra></extra>"
    )
    fig_scatter.update_layout(height=500)  


    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Price vs Area")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader(" Bedroom Distribution")
        df2['locality'] = df2['locality'].astype(str).str.strip().str.lower()
        localities = sorted(df2['locality'].dropna().unique().tolist())
        localities = ["overall"] + localities

        selected_loc = st.selectbox(
            "Select Locality",
            localities
        )

    
        

        
        filtered_df = df2[df2['locality'] == selected_loc]

        if selected_loc == "overall":
            filtered_df = df
        else:
            filtered_df = df[df['locality'] == selected_loc]

        if filtered_df.empty:
            st.warning("No data available")

        else:
            bedroom_counts = filtered_df['bedrooms'].value_counts().reset_index()
            bedroom_counts.columns = ['bedrooms', 'count']

            fig_pie = px.pie(
                bedroom_counts,
                names='bedrooms',
                values='count',
                title=f"{selected_loc}"
            )

            fig_pie.update_traces(
                
                textinfo='percent+label',
                hovertemplate=
                "Bedrooms: %{label}<br>" +
                "Count: %{value}<br>" +
                "Share: %{percent}<extra></extra>"
            )
            
            fig_pie.update_layout(height=500)


            st.plotly_chart(fig_pie, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(" price per bhk")
        df_bhk = df2[df2['bedrooms'].isin([1,2,3])]

        fig_box = px.box(
            df_bhk,
            x='bedrooms',
            y='price',
            title="Price Range by BHK Type",
            labels={
                'bedrooms': 'BHK',
                'price': 'Price (Cr)'
            }
            

        )
        fig_box.update_traces(
            width=0.5,
            marker=dict(size=4)
        )

        fig_box.update_layout(height=500)

        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        st.subheader("🏆 Top Localities by Price per Sqft")


        top_loc = (
            df2.groupby('locality')['price_sqft']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

      
        fig_bar = px.bar(
            top_loc,
            x='price_sqft',
            y='locality',
            orientation='h',   
            text='price_sqft',
            color='price_sqft',
            color_continuous_scale='Blues'
        )

        fig_bar.update_traces(width=0.8)

        
        fig_bar.update_layout(
            height=500,
            bargap=0,
            bargroupgap=0,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Price per Sqft (₹)",
            yaxis_title="",
           
            margin=dict(l=0, r=0, t=30, b=0)
        )

      
        fig_bar.update_traces(
            width=0.5,
            texttemplate='%{text:.0f}',
            textposition='outside'
        )

        st.plotly_chart(fig_bar, use_container_width=True)
    
    col5, col6 = st.columns(2)


    with col5:
        st.subheader(" Price Distribution")

        fig_hist = px.histogram(
            df2,
            x='price',
            nbins=30,
            labels={'price': 'Price (Cr)'}
        )


        fig_hist.update_traces(
            xbins=dict(
                start=0,   
                end=20,
                size=1     
            )
        )
        fig_hist.update_layout(
            height=400,
            bargap=0.2,   
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col6:
        st.subheader(" Top 10 Societies (Avg Price)")

        top_soc = (
            df2.groupby(['society_name', 'locality'])['price']
            .mean()
            .reset_index()
            .sort_values(by='price', ascending=False)
            .head(10)
        )

        
        top_soc['price'] = top_soc['price'].round(2)

        

        st.dataframe(
            top_soc.style.format({
                "price": "{:.2f} Cr"
            }),
            use_container_width=True
        )


elif page == 'Insight module':

    

    st.title(" Insight Module")
    st.caption("What-if simulator: Change one feature and see price impact")

    # ---- LOAD MODEL ----
    model = joblib.load("xgb_pipeline.joblib")

    # ==============================
    # 🔹 USER INPUT
    # ==============================
    st.markdown("### 🔹 Select Your Flat")

    col1, col2, col3 = st.columns(3)

    with col1:
        locality = st.selectbox("Locality", sorted(df["locality"].dropna().unique()))
        locality = str(locality).strip()

    with col2:
        bedrooms = int(st.selectbox("Bedrooms", sorted(df["bedrooms"].unique())))

    with col3:
        bathrooms = int(st.selectbox("Bathrooms", sorted(df["bathrooms"].unique())))

    col4, col5, col6 = st.columns(3)

    with col4:
        furnish_label = st.selectbox(
            "Furnishing",
            ["Unfurnished", "Semi Furnished", "Fully Furnished"]
        )

    with col5:
        floor_label = st.selectbox(
            "Floor Category",
            ["Lower", "Middle", "Upper"]
        )

    with col6:
        age_label = st.selectbox(
            "Age Category",
            ["Very Old", "Old", "Moderate", "New"],
            index=2
        )

    buildup = float(st.slider(
        "Build-up Area",
        int(df["buildup_area"].min()),
        int(df["buildup_area"].max()),
        800
    ))

   
    carpet = buildup * 0.8
    total_floors = 10


    st.markdown("### Your Selected Flat")

    s1, s2, s3= st.columns(3)
    s4,s5,s6=st.columns(3)

    s1.metric(" Locality", locality)
    s2.metric(" Bedrooms", f"{bedrooms} BHK")
    s3.metric(" Bathrooms", bathrooms)
    s4.metric(" Area", f"{int(buildup)} sq.ft")
    s5.metric(" Furnish", furnish_label)
    s6.metric(" Floor", floor_label)

    base = pd.DataFrame([{
        "buildup_area": buildup,
        "carpet_area": carpet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "furnish_detail": furnish_label,
        "age_category": age_label,
        "floor_category": floor_label,
        "total_floors": total_floors,
        "locality": locality
    }])

    base_price = model.predict(base)[0]

    
    st.markdown("###  Change One Thing")

    st.info("ℹ️ Only one feature is changed at a time. Others remain constant.")

    feature = st.selectbox(
        "What do you want to change?",
        ["locality", "furnish_detail", "floor_category"]
    )

    if feature == "locality":
        new_value = st.selectbox("New Locality", sorted(df["locality"].dropna().unique()))
        new_value = str(new_value).strip()

    elif feature == "furnish_detail":
        new_value = st.selectbox(
            "New Furnishing",
            ["Unfurnished", "Semi Furnished", "Fully Furnished"]
        )

    elif feature == "floor_category":
        new_value = st.selectbox(
            "New Floor",
            ["Lower", "Middle", "Upper"]
        )
    temp = base.copy()
    temp[feature] = new_value

    new_price = model.predict(temp)[0]

    change = new_price - base_price
    percent = (change / base_price) * 100

  
    st.markdown("###  Price Comparison")

    c1, c2, c3 = st.columns(3)

    c1.metric("Original Price", f"₹ {base_price:.2f} Cr")
    c2.metric("New Price", f"₹ {new_price:.2f} Cr")
    c3.metric("Change", f"{change:.2f} Cr", f"{percent:.1f}%")

    
    st.markdown("###  Why did price change?")

    if feature == "locality":
        st.success(" Location has the highest impact on price.")
        st.write(f"{new_value} is likely a higher demand or premium area.")

    elif feature == "furnish_detail":
        st.write(" Better furnishing increases perceived property value.")

    elif feature == "floor_category":
        st.write(" Floor level affects view, sunlight, and desirability.")

    if change > 0:
        st.success(f"Price increased by ₹ {abs(change):.2f} Cr ({percent:.1f}%)")
    else:
        st.error(f"Price decreased by ₹ {abs(change):.2f} Cr ({percent:.1f}%)")

    
    st.markdown("###  Visual Explanation")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[base_price],
        y=["Original Price"],
        orientation='h',
        marker=dict(color="#4a90e2")
    ))

    fig.add_trace(go.Bar(
        x=[new_price],
        y=["New Price"],
        orientation='h',
        marker=dict(color="#2ecc71")
    ))

    fig.update_layout(
        barmode='group',
        xaxis_title="Price (Cr)",
        height=250
    )

    st.plotly_chart(fig, use_container_width=True)



# load once


ACCESS_KEY = "JltbxKqI8HZTSeIyBSZ3W8oR1JYx4GXt0EtRbsS7h70"

@st.cache_data(show_spinner=False)
def get_image(query_list):
    url = "https://api.unsplash.com/search/photos"

    for query in query_list:
        params = {
            "query": query,
            "client_id": ACCESS_KEY,
            "per_page": 1
        }

        res = requests.get(url, params=params)

        if res.status_code != 200:
            continue

        data = res.json()

        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["urls"]["regular"]

    return None

if page=='Recommendation':

    df3=pd.read_csv(r"C:\Users\DELL\Desktop\house price predicton\flats_final")
    st.title(" Society Recommendation")

    st.markdown("###  Find Best Societies for You")

    col1, col2 = st.columns(2)

    with col1:
        locality = st.selectbox(
            "Select Locality",
            sorted(df3["locality"].dropna().unique())
        )

    with col2:
        sqft = st.slider(
            "Select Area (sq.ft)",
            int(df3["buildup_area"].min()),
            int(df3["buildup_area"].max()),
            800
        )
    filtered = df3[
    (df3["locality"] == locality) &
    (df3["buildup_area"] >= sqft * 0.8) &
    (df3["buildup_area"] <= sqft * 1.2)]

    if filtered.empty:
        st.warning("⚠️ No similar flats found. Try changing area or locality.")
        st.stop()

    grouped = filtered.groupby("society_name").agg({
        "price": "mean",
        "buildup_area": "mean",
        "bedrooms": "mean"
    }).reset_index()

    grouped["count"] = filtered.groupby("society_name").size().values

    def calculate_score(row):
        # area similarity (closer = better)
        area_score = 1 - abs(row["buildup_area"] - sqft) / sqft

        # price preference (lower is better)
        price_score = 1 / row["price"]

        # availability (more listings = more reliable)
        count_score = row["count"] / grouped["count"].max()

        # final weighted score
        return (0.5 * area_score) + (0.3 * price_score) + (0.2 * count_score)


    grouped["score"] = grouped.apply(calculate_score, axis=1)
    grouped = grouped.sort_values("score", ascending=False)
    top5 = grouped.head(5)
    
    st.markdown("##  Best Societies for You")


    for _, row in top5.iterrows():

        col1, col2 = st.columns([1, 2])

        # ---- IMAGE (placeholder for now) ----
        with col1:
            
            queries = [
                f"{row['society_name']} building",
                f"{locality} apartment building",
                "modern apartment building exterior",
                "luxury apartment india"
            ]

            img_url = get_image(queries)

            if img_url:
                st.image(img_url, use_container_width=True)
                st.caption("📷 Representative image (not exact society)")
            else:
                st.info("📷 No image found, showing placeholder")
                st.image("https://via.placeholder.com/300", use_container_width=True)
                    

        # ---- DETAILS ----
        with col2:
            st.markdown(f"###  {row['society_name']}")
            st.write(f" {locality}")
            st.write(f" Avg Price: ₹ {row['price']:.2f} Cr")
            st.write(f" Avg Area: {int(row['buildup_area'])} sq.ft")
            st.write(f" Listings Available: {int(row['count'])}")

            # ---- SMART TAG ----
            if row["score"] > grouped["score"].mean():
                st.success(" Best Value")
            else:
                st.info(" Average Option")

        st.markdown("---")
            




