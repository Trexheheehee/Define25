import os
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import traceback
import warnings

# Suppress pandas warnings about datetime format inference
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# Aircraft Tail Assignment Environment
class TailAssignmentEnv:
    def __init__(self, flights_df, aircraft_df):
        self.flights = self._preprocess_flights(flights_df.copy())
        self.aircraft = self._preprocess_aircraft(aircraft_df.copy())
        self.current_step = 0
        self.assignments = {}
        self.aircraft_availability = {}
        
    def _preprocess_flights(self, df):
        # Convert datetime strings to datetime objects with flexible format handling
        if 'Scheduled Time of Departure' in df.columns:
            # Try multiple formats in order
            formats = [
                '%Y-%m-%d %H:%M:%S',  # 2023-01-01 14:30:00
                '%Y-%m-%d %H:%M',     # 2023-01-01 14:30
                '%d/%m/%Y %H:%M',     # 01/01/2023 14:30
                '%m/%d/%Y %H:%M',     # 01/01/2023 14:30
                '%Y-%m-%dT%H:%M:%S',  # ISO format
                '%Y%m%d%H%M'          # Compact format
            ]
            
            # Show sample of date format for debugging
            st.sidebar.write("Sample departure time format:", df['Scheduled Time of Departure'].iloc[0] if len(df) > 0 else "No data")
            
            # Try each format
            df['Scheduled Time of Departure'] = self._parse_datetime_column(df['Scheduled Time of Departure'], formats)
            
        if 'Schedules Time of Arrival' in df.columns:
            df['Schedules Time of Arrival'] = self._parse_datetime_column(df['Schedules Time of Arrival'], formats)
        
        # Sort flights by departure time
        if 'Scheduled Time of Departure' in df.columns:
            df = df.sort_values('Scheduled Time of Departure')
        
        return df
    
    def _parse_datetime_column(self, column, formats):
        # Try each format until one works
        for fmt in formats:
            try:
                return pd.to_datetime(column, format=fmt, errors='coerce')
            except:
                continue
        
        # If none worked, fall back to letting pandas guess
        return pd.to_datetime(column, errors='coerce')
        
    def _preprocess_aircraft(self, df):
        # Ensure aircraft registration is the index
        if 'Aircraft Registration' in df.columns:
            df = df.set_index('Aircraft Registration')
        return df
    
    def reset(self):
        self.current_step = 0
        self.assignments = {}
        self.aircraft_availability = {reg: None for reg in self.aircraft.index}
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.flights):
            return None
        
        flight = self.flights.iloc[self.current_step]
        available_aircraft = self._get_available_aircraft(flight)
        
        return {
            'flight': flight,
            'available_aircraft': available_aircraft,
            'step': self.current_step
        }
    
    def _get_available_aircraft(self, flight):
        available = []
        
        # Check aircraft type compatibility
        required_type = flight.get('Aircraft Type', None)
        required_capacity = flight.get('Physical Seating capacity', 0)
        
        for reg, last_available in self.aircraft_availability.items():
            aircraft = self.aircraft.loc[reg]
            
            # Type compatibility check
            type_match = True
            if required_type is not None and 'Aircraft Type' in aircraft:
                type_match = aircraft['Aircraft Type'] == required_type
            
            # Capacity check
            capacity_match = True
            if required_capacity > 0 and 'Seating capacity' in aircraft:
                capacity_match = aircraft['Seating capacity'] >= required_capacity
            
            # Availability check
            time_match = True
            if last_available is not None and 'Scheduled Time of Departure' in flight:
                min_ground_time = flight.get('Minimum Ground Time', 30)  # Default 30 minutes
                required_available = flight['Scheduled Time of Departure'] - pd.Timedelta(minutes=min_ground_time)
                time_match = last_available <= required_available
            
            if type_match and capacity_match and time_match:
                available.append(reg)
        
        return available
    
    def step(self, action):
        if self.current_step >= len(self.flights):
            return None, 0, True, {}
        
        flight = self.flights.iloc[self.current_step]
        flight_id = flight.get('Flight Identifier', f"Flight_{self.current_step}")
        
        # Get available aircraft
        available_aircraft = self._get_available_aircraft(flight)
        
        # Check if action is valid
        if action not in available_aircraft:
            reward = -100  # Penalty for invalid assignment
            self.assignments[flight_id] = None
        else:
            # Valid assignment
            aircraft_reg = action
            self.assignments[flight_id] = aircraft_reg
            
            # Update aircraft availability
            if 'Schedules Time of Arrival' in flight:
                self.aircraft_availability[aircraft_reg] = flight['Schedules Time of Arrival']
            
            # Calculate reward
            reward = self._calculate_reward(flight, aircraft_reg)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.flights)
        
        return self._get_state(), reward, done, {'assignments': self.assignments}
    
    def _calculate_reward(self, flight, aircraft_reg):
        aircraft = self.aircraft.loc[aircraft_reg]
        
        # Base reward for successful assignment
        reward = 10
        
        # Type matching bonus
        if 'Aircraft Type' in flight and 'Aircraft Type' in aircraft:
            if flight['Aircraft Type'] == aircraft['Aircraft Type']:
                reward += 5
        
        # Capacity utilization reward
        if 'Physical Seating capacity' in flight and 'Seating capacity' in aircraft:
            capacity_diff = aircraft['Seating capacity'] - flight['Physical Seating capacity']
            if capacity_diff >= 0:
                # Efficient use of capacity (not too much waste)
                efficiency = 1.0 - (capacity_diff / aircraft['Seating capacity'])
                reward += 5 * efficiency
            else:
                # Penalty for insufficient capacity
                reward -= 10
        
        return reward

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = {}
    
    def _get_state_key(self, state):
        if state is None:
            return "terminal"
        
        flight = state['flight']
        flight_id = flight.get('Flight Identifier', f"Flight_{state['step']}")
        return f"{flight_id}_{state['step']}"
    
    def _get_q_value(self, state, action):
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            
        return self.q_table[state_key][action]
    
    def _update_q_value(self, state, action, reward, next_state):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state_key in self.q_table and next_state_key != "terminal":
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
            
        # Q-learning update
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
    
    def choose_action(self, state):
        if state is None:
            return None
            
        available_aircraft = state['available_aircraft']
        if not available_aircraft:
            return None
            
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(available_aircraft)
            
        # Exploitation: best action based on Q-values
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            
        # Find best action among available aircraft
        best_action = None
        best_value = float('-inf')
        
        for action in available_aircraft:
            q_value = self._get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
                
        # If no best action found, choose randomly
        if best_action is None:
            best_action = random.choice(available_aircraft)
            
        return best_action
    
    def train(self, episodes, callback=None):
        rewards_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                
                if action is None:
                    # No valid action available, skip to next flight
                    next_state, reward, done, _ = self.env.step(None)
                else:
                    next_state, reward, done, _ = self.env.step(action)
                    
                # Update Q-values
                if action is not None:
                    self._update_q_value(state, action, reward, next_state)
                    
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Decay exploration rate
            self.epsilon *= self.epsilon_decay
            
            # Record rewards
            rewards_history.append(total_reward)
            
            # Callback for progress updates
            if callback and episode % 10 == 0:
                callback(episode, episodes, total_reward)
        
        return rewards_history, self.env.assignments

def main():
    try:
        st.title("Aircraft Tail Assignment System")
        
        # File uploads
        col1, col2 = st.columns(2)
        with col1:
            flights_file = st.file_uploader("Upload Flights Data (CSV)", type="csv", key='flights')
        with col2:
            aircraft_file = st.file_uploader("Upload Aircraft Data (CSV)", type="csv", key='aircraft')
        
        if flights_file is not None and aircraft_file is not None:
            try:
                flights_df = pd.read_csv(flights_file)
                aircraft_df = pd.read_csv(aircraft_file)
                
                # Show previews
                st.subheader("Data Preview")
                st.write("Flights Data:", flights_df.head())
                st.write("Aircraft Data:", aircraft_df.head())
                
                # Initialize environment
                env = TailAssignmentEnv(flights_df, aircraft_df)
                
                # Training parameters
                st.subheader("Training Parameters")
                episodes = st.slider("Number of Episodes", 10, 1000, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
                discount_factor = st.slider("Discount Factor", 0.5, 0.99, 0.9)
                exploration_rate = st.slider("Initial Exploration Rate", 0.1, 1.0, 1.0)
                
                # Initialize agent
                agent = QLearningAgent(
                    env, 
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    exploration_rate=exploration_rate
                )
                
                # Training
                if st.button("Train Model"):
                    progress_bar = st.progress(0)
                    reward_placeholder = st.empty()
                    
                    def update_progress(episode, total_episodes, reward):
                        progress = (episode + 1) / total_episodes
                        progress_bar.progress(progress)
                        reward_placeholder.text(f"Episode {episode+1}/{total_episodes} - Reward: {reward:.2f}")
                    
                    with st.spinner("Training in progress..."):
                        rewards, assignments = agent.train(episodes, callback=update_progress)
                    
                    st.success("Training completed!")
                    
                    # Show results
                    st.subheader("Training Results")
                    
                    # Plot rewards
                    rewards_df = pd.DataFrame(rewards, columns=["Reward"])
                    st.line_chart(rewards_df)
                    
                    # Show assignments
                    st.subheader("Aircraft Assignments")
                    assignments_data = []
                    
                    for flight_idx, flight in flights_df.iterrows():
                        flight_id = flight.get('Flight Identifier', f"Flight_{flight_idx}")
                        assigned_aircraft = assignments.get(flight_id, "Not assigned")
                        
                        assignments_data.append({
                            "Assigned Aircraft": assigned_aircraft,
                            "Flight1": f"Flight ID: {flight_id}\nDeparture: {flight.get('Scheduled Time of Departure', 'Unknown')}\nFrom: {flight.get('Departure Station', 'Unknown')}\nTo: {flight.get('Arrival Station', 'Unknown')}",
                            "Flight2": f"Flight ID: {flight_id}\nDeparture: {flight.get('Scheduled Time of Departure', 'Unknown')}\nFrom: {flight.get('Departure Station', 'Unknown')}\nTo: {flight.get('Arrival Station', 'Unknown')}",
                            "Flight3": f"Flight ID: {flight_id}\nDeparture: {flight.get('Scheduled Time of Departure', 'Unknown')}\nFrom: {flight.get('Departure Station', 'Unknown')}\nTo: {flight.get('Arrival Station', 'Unknown')}"
                        })
                    assignments_df = pd.DataFrame(assignments_data)
                    st.dataframe(assignments_df)
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.error(traceback.format_exc())
        else:
            st.info("Please upload both flight and aircraft data files to begin.")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()