import streamlit as st
import requests
from datetime import date
import openai
import anthropic
import google.generativeai as genai

# ============================================
# Lab 5a: Weather & Distance Functions
# ============================================

def get_weather_data(city, api_key):
    """
    Fetch weather data from OpenWeatherMap API
    Returns: dict with temperature, feels_like, humidity, description, wind_speed
    """
    try:
        # Convert city format for OpenWeatherMap API
        # "Syracuse, NY" -> "Syracuse,US"
        # "London, England" -> "London,GB"
        city_mapping = {
            "Syracuse, NY": "Syracuse,US",
            "New York, NY": "New York,US",
            "Miami, FL": "Miami,US",
            "Los Angeles, CA": "Los Angeles,US",
            "San Diego, CA": "San Diego,US",
            "Seattle, WA": "Seattle,US",
            "Chicago, IL": "Chicago,US",
            "London, England": "London,GB",
            "Paris, France": "Paris,FR",
            "Cancun, Mexico": "Cancun,MX",
            "Tokyo, Japan": "Tokyo,JP",
            "Seoul, South Korea": "Seoul,KR"
        }
        
        # Use mapped city name if available, otherwise use original
        query_city = city_mapping.get(city, city)
        
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": query_city,
            "appid": api_key,
            "units": "metric"  # Get temperature in Celsius
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        weather_data = {
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }
        
        return weather_data
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data for {city}: {str(e)}")
        st.info(f"💡 Try using format like: 'Syracuse' or 'New York' or 'London'")
        return None
    except KeyError as e:
        st.error(f"Error parsing weather data for {city}: {str(e)}")
        return None


def calculate_travel_info(origin, destination, provider, model):
    """
    Calculate travel information between two cities using LLM
    Returns: dict with estimated_distance_km, estimated_drive_time_hours, estimated_flight_time_hours
    """
    prompt = f"""You are a travel distance and time estimation expert. Provide accurate travel information between these two cities:

Origin: {origin}
Destination: {destination}

Please provide ONLY a JSON response with the following structure (no other text):
{{
    "estimated_distance_km": <number>,
    "estimated_drive_time_hours": <number or null if not driveable>,
    "estimated_flight_time_hours": <number>,
    "is_international": <true or false>,
    "notes": "<any important travel notes>"
}}

Important:
- Use realistic distances and times
- Set drive time to null if cities are on different continents or separated by ocean
- Flight time should include typical flight duration (not including airport time)
- Be accurate based on real geography"""

    try:
        response = llm_call(provider, model, prompt, temperature=0.3)
        
        if not response:
            return get_fallback_travel_info()
        
        # Try to extract JSON from response
        import json
        import re
        
        # Remove markdown code blocks if present
        json_match = re.search(r'(?:json)?\s*(\{.*?\})\s*', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        data = json.loads(json_str)
        
        return {
            "estimated_distance_km": data.get("estimated_distance_km", 1000),
            "estimated_drive_time_hours": data.get("estimated_drive_time_hours"),
            "estimated_flight_time_hours": data.get("estimated_flight_time_hours", 3),
            "is_international": data.get("is_international", False),
            "notes": data.get("notes", "")
        }
    
    except Exception as e:
        st.warning(f"Could not parse LLM response for travel info. Using fallback estimates. Error: {str(e)}")
        return get_fallback_travel_info()


def get_fallback_travel_info():
    """Fallback travel info if LLM fails"""
    return {
        "estimated_distance_km": 1000,
        "estimated_drive_time_hours": 10,
        "estimated_flight_time_hours": 3,
        "is_international": False,
        "notes": "Estimated values (LLM unavailable)"
    }


# ============================================
# Lab 5b: LLM Helper Function
# ============================================

def llm_call(provider, model, prompt, temperature=0.7):
    """
    Universal LLM caller that works with OpenAI, Claude, or Gemini
    Returns: string response from the model
    """
    try:
        if provider == "openai":
            # OpenAI API call
            api_key = st.secrets["OPENAI_API_KEY"]
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        
        elif provider == "claude":
            # Anthropic Claude API call
            api_key = st.secrets["ANTHROPIC_API_KEY"]
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif provider == "gemini":
            # Google Gemini API call
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text
        
        else:
            return f"Error: Unknown provider '{provider}'"
    
    except Exception as e:
        st.error(f"Error calling {provider} API: {str(e)}")
        return None


# ============================================
# Lab 5c: Multi-Agent System
# ============================================

def weather_comparison_agent(origin_weather, dest_weather, origin_name, dest_name):
    """
    Agent 1: Weather Comparison Agent (No LLM - just formatting)
    Compares weather between origin and destination
    """
    if not origin_weather or not dest_weather:
        return "Unable to compare weather data."
    
    temp_diff = dest_weather["temperature"] - origin_weather["temperature"]
    
    comparison = f"""
## 🌤 Weather Comparison

*{origin_name}:*
- Temperature: {origin_weather['temperature']:.1f}°C (feels like {origin_weather['feels_like']:.1f}°C)
- Conditions: {origin_weather['description'].title()}
- Humidity: {origin_weather['humidity']}%
- Wind Speed: {origin_weather['wind_speed']} m/s

*{dest_name}:*
- Temperature: {dest_weather['temperature']:.1f}°C (feels like {dest_weather['feels_like']:.1f}°C)
- Conditions: {dest_weather['description'].title()}
- Humidity: {dest_weather['humidity']}%
- Wind Speed: {dest_weather['wind_speed']} m/s

*Temperature Difference:* {abs(temp_diff):.1f}°C {"warmer" if temp_diff > 0 else "cooler"} at destination
"""
    
    # Add weather warnings
    warnings = []
    if dest_weather['temperature'] < 0:
        warnings.append("⚠ Freezing temperatures at destination - pack warm!")
    if dest_weather['temperature'] > 35:
        warnings.append("⚠ Very hot at destination - stay hydrated!")
    if "rain" in dest_weather['description'].lower():
        warnings.append("🌧 Rain expected at destination - bring an umbrella!")
    if "snow" in dest_weather['description'].lower():
        warnings.append("❄ Snow at destination - winter gear required!")
    
    if warnings:
        comparison += "\n*Alerts:*\n" + "\n".join(warnings)
    
    return comparison


def travel_logistics_agent(provider, model, origin, destination, travel_info, departure_date):
    """
    Agent 2: Travel Logistics Agent (Uses LLM)
    Provides transportation recommendations and travel tips
    """
    distance = travel_info["estimated_distance_km"]
    drive_time = travel_info["estimated_drive_time_hours"]
    flight_time = travel_info["estimated_flight_time_hours"]
    is_international = travel_info.get("is_international", False)
    notes = travel_info.get("notes", "")
    
    drive_info = f"{drive_time} hours" if drive_time else "Not recommended (overseas/too far)"
    international_note = "\n- This is an INTERNATIONAL trip - don't forget your passport!" if is_international else ""
    
    prompt = f"""You are a travel logistics expert. Provide practical travel advice for this trip:

Origin: {origin}
Destination: {destination}
Distance: {distance} km
Driving time: {drive_info}
Flight time: {flight_time} hours
Departure date: {departure_date}
{f'Additional notes: {notes}' if notes else ''}
{international_note}

Please provide:
1. Recommended mode of transportation (drive vs. fly) with reasoning
2. Best departure time considerations
3. Estimated arrival time
4. Important travel tips (traffic, airport, border crossings, etc.)
5. Any cost considerations

Keep your response concise and practical (200-300 words)."""
    
    response = llm_call(provider, model, prompt)
    return response if response else "Unable to generate travel logistics recommendations."


def packing_advisor_agent(provider, model, weather_comparison, trip_duration_days):
    """
    Agent 3: Packing Advisor Agent (Uses LLM)
    Suggests what to pack based on weather and trip duration
    """
    prompt = f"""You are a packing expert. Based on the weather comparison below and trip duration, suggest what to pack.

Weather Information:
{weather_comparison}

Trip Duration: {trip_duration_days} days

Please provide:
1. Essential clothing items for the weather conditions
2. Accessories needed (umbrella, sunglasses, etc.)
3. 5-7 specific items with brief reasoning
4. Any special considerations

Keep your response organized and concise (200-250 words)."""
    
    response = llm_call(provider, model, prompt)
    return response if response else "Unable to generate packing recommendations."


def activity_planner_agent(provider, model, destination, dest_weather, trip_duration_days):
    """
    Agent 4: Activity Planner Agent (Uses LLM)
    Suggests activities and creates an itinerary based on destination and weather
    """
    prompt = f"""You are a travel activity planner. Suggest activities for this trip:

Destination: {destination}
Current Weather: {dest_weather['temperature']:.1f}°C, {dest_weather['description']}
Trip Duration: {trip_duration_days} days

Please provide:
1. 3-5 activities suitable for the current weather
2. Indoor alternatives if weather is poor
3. Local food/restaurant recommendations
4. A brief day-by-day itinerary outline
5. Any local events or seasonal attractions

Keep your response engaging and practical (250-300 words)."""
    
    response = llm_call(provider, model, prompt)
    return response if response else "Unable to generate activity recommendations."


# ============================================
# Lab 5d: Streamlit Interface
# ============================================

def main():
    st.set_page_config(page_title="Travel Planning Assistant", page_icon="🌍", layout="wide")
    
    st.title("🌍 Lab 5: Multi-Agent Travel Planning Assistant")
    st.markdown("Plan your perfect trip with AI-powered agents analyzing weather, logistics, packing, and activities!")
    
    # Sidebar for LLM configuration
    st.sidebar.header("🤖 LLM Configuration")
    
    provider = st.sidebar.selectbox(
        "Select LLM Provider",
        ["openai", "claude", "gemini"]
    )
    
    # Model selection based on provider
    if provider == "openai":
        model = st.sidebar.selectbox(
            "Select Model",
            ["gpt-5-nano", "gpt-5-chat-latest"]
        )
    elif provider == "claude":
        model = st.sidebar.selectbox(
            "Select Model",
            ["claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"]
        )
    else:  # gemini
        model = st.sidebar.selectbox(
            "Select Model",
            ["gemini-2.5-pro", "gemini-2.5-flash"]
        )
    
    st.sidebar.info(f"*Current Selection:*\n\n🔧 Provider: {provider}\n\n🤖 Model: {model}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 How it works:")
    st.sidebar.markdown("""
    1. *Weather Agent* - Compares weather
    2. *Logistics Agent* - Travel recommendations
    3. *Packing Agent* - What to bring
    4. *Activity Agent* - Things to do
    """)
    
    # Main input section
    st.header("📍 Trip Details")
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origin City", value="Syracuse, NY", help="e.g., 'New York, NY' or 'London, England'")
    with col2:
        destination = st.text_input("Destination City", value="Miami, FL", help="e.g., 'Miami, FL' or 'Paris, France'")
    
    col3, col4 = st.columns(2)
    with col3:
        departure_date = st.date_input("Departure Date", value=date.today())
    with col4:
        trip_duration = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=3)
    
    run_button = st.button("🚀 Plan My Trip", type="primary", use_container_width=True)
    
    # Execute when button is clicked
    if run_button:
        if not origin or not destination:
            st.error("Please enter both origin and destination cities.")
            return
        
        # Get API key for weather
        try:
            weather_api_key = st.secrets["api_key"]
        except:
            st.error("Weather API key not found in secrets. Please add 'api_key' to your secrets.toml file.")
            return
        
        with st.status("🔄 Planning your trip...", expanded=True) as status:
            # Step 1: Fetch weather data
            st.write("☁ Fetching weather data...")
            origin_weather = get_weather_data(origin, weather_api_key)
            dest_weather = get_weather_data(destination, weather_api_key)
            
            if not origin_weather or not dest_weather:
                status.update(label="❌ Failed to fetch weather data", state="error")
                return
            
            # Step 2: Calculate travel info
            st.write("🗺 Calculating travel information...")
            travel_info = calculate_travel_info(origin, destination, provider, model)
            
            # Step 3: Run Agent 1 - Weather Comparison
            st.write("🌤 Comparing weather conditions...")
            weather_summary = weather_comparison_agent(origin_weather, dest_weather, origin, destination)
            
            # Step 4: Run Agent 2 - Travel Logistics
            st.write("✈ Generating travel recommendations...")
            logistics = travel_logistics_agent(provider, model, origin, destination, travel_info, str(departure_date))
            
            # Step 5: Run Agent 3 - Packing Advisor
            st.write("🎒 Creating packing list...")
            packing = packing_advisor_agent(provider, model, weather_summary, trip_duration)
            
            # Step 6: Run Agent 4 - Activity Planner
            st.write("🗺 Planning activities...")
            activities = activity_planner_agent(provider, model, destination, dest_weather, trip_duration)
            
            status.update(label="✅ Trip planning complete!", state="complete")
        
        # Display results
        st.success("🎉 Your personalized travel plan is ready!")
        
        # Section 1: Trip Overview
        st.header("📋 Trip Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📍 Route", f"{origin} → {destination}")
        with col2:
            st.metric("📏 Distance", f"{travel_info['estimated_distance_km']} km")
        with col3:
            st.metric("📅 Duration", f"{trip_duration} days")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            if travel_info['estimated_drive_time_hours']:
                st.metric("🚗 Drive Time", f"{travel_info['estimated_drive_time_hours']:.1f} hrs")
            else:
                st.metric("🚗 Drive Time", "N/A")
        with col5:
            st.metric("✈ Flight Time", f"{travel_info['estimated_flight_time_hours']:.1f} hrs")
        with col6:
            st.metric("📆 Departure", departure_date.strftime("%b %d, %Y"))
        
        # Section 2: Weather Comparison
        with st.expander("☁ *Weather Comparison*", expanded=True):
            st.markdown(weather_summary)
        
        # Section 3: Travel Logistics
        with st.expander("✈ *Travel Recommendations*", expanded=True):
            st.markdown(logistics)
        
        # Section 4: Packing List
        with st.expander("🎒 *Packing List*", expanded=True):
            st.markdown(packing)
        
        # Section 5: Activity Itinerary
        with st.expander("🗺 *Activity Itinerary*", expanded=True):
            st.markdown(activities)
        
        # Footer with download option
        st.markdown("---")
        st.info("💡 *Tip:* Screenshot or copy this information for your trip planning!")


if __name__ == "__main__":
    main()