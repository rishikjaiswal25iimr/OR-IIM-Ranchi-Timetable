import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ortools.sat.python import cp_model
import math
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="IIM Ranchi Timetable Optimizer", page_icon="📅")

# ==========================================
# 1. SIDEBAR: INDEPENDENT VARIABLES
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f6/Indian_Institute_of_Management_Ranchi_Logo.svg/1200px-Indian_Institute_of_Management_Ranchi_Logo.svg.png", width=150)
st.sidebar.header("⚙️ Scheduling Parameters")

max_capacity = st.sidebar.slider("Max Students per Section", 40, 100, 70, help="Courses with more students will be split into Section A & B.")
initial_rooms = st.sidebar.number_input("Initial Classrooms (Weeks 1-4)", min_value=1, max_value=20, value=10)
reduced_rooms = st.sidebar.number_input("Reduced Classrooms (Weeks 5-10)", min_value=1, max_value=20, value=4)
reduction_week = st.sidebar.slider("Week of Capacity Reduction", 1, 10, 5)
max_daily_sessions_student = st.sidebar.slider("Max Daily Sessions per Section", 1, 5, 2, help="To prevent student burnout.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Hard Constraints Enforced:**
- 20 Sessions per course section
- Zero faculty overlapping
- Zero student overlapping
- Strict classroom capacity limits per week
- Continuous slots without lunch break
- Reduced Sunday timings
""")

# ==========================================
# 2. DATA PROCESSING (EXCEL PARSER)
# ==========================================
@st.cache_data
def process_uploaded_excel(uploaded_file, max_cap):
    courses_raw = []
    
    # Read the Excel workbook
    xls = pd.ExcelFile(uploaded_file)
    
    # Parse each sheet robustly
    for sheet_name in xls.sheet_names:
        # Load sheet without headers so we can find where the actual data starts
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        
        faculty_name = "Unknown Faculty"
        course_name = "Unknown Course"
        header_idx = -1
        
        for i, row in df_sheet.iterrows():
            # Convert row to list of strings, treating NaN as empty string
            cols = [str(val).strip() if pd.notna(val) else "" for val in row.values]
            
            if len(cols) > 1 and "Faculty Name" in cols[0]:
                faculty_name = cols[1] if cols[1] else "Unknown Faculty"
                
            # Find the header row for students (looking for SN, Serial No., or Student ID)
            if "SN" in cols[0] or "Serial No." in cols[0] or "Student ID" in cols:
                header_idx = i
                # Look backwards from the header to find the course name
                for j in range(i-1, -1, -1):
                    prev_cols = [str(val).strip() if pd.notna(val) else "" for val in df_sheet.iloc[j].values]
                    if prev_cols and prev_cols[0]:
                        c_name = prev_cols[0]
                        if "Group Mail ID" not in c_name and "Faculty Name" not in c_name:
                            course_name = c_name
                            break
                break
                
        # Extract students
        students = []
        if header_idx != -1:
            headers = [str(val).strip() if pd.notna(val) else "" for val in df_sheet.iloc[header_idx].values]
            if "Student ID" in headers:
                sid_idx = headers.index("Student ID")
                for i in range(header_idx + 1, len(df_sheet)):
                    val = df_sheet.iloc[i, sid_idx]
                    if pd.notna(val) and str(val).strip():
                        students.append(str(val).strip())
                        
        courses_raw.append({
            "Course": course_name,
            "Faculty": faculty_name,
            "Students_List": students,
            "Total_Students": len(students)
        })
        
    df = pd.DataFrame(courses_raw)
    
    # Split sections and assign specific students to each section
    sections_list = []
    for _, row in df.iterrows():
        students = row['Students_List']
        num_sections = 2 if row['Total_Students'] > max_cap else 1
        
        # Split students as evenly as possible
        section_size = math.ceil(len(students) / num_sections) if num_sections > 0 else 0
        
        for sec in range(num_sections):
            sec_name = "Sec A" if sec == 0 else "Sec B"
            if num_sections == 1: sec_name = "Core"
            
            sec_students = students[sec * section_size : (sec + 1) * section_size]
            
            sections_list.append({
                "Course": row['Course'],
                "Section": sec_name,
                "Faculty": row['Faculty'],
                "Students_Count": len(sec_students),
                "Student_IDs": set(sec_students) # Stored as a Set for fast intersection/overlap math
            })
            
    return df, pd.DataFrame(sections_list)


# ==========================================
# 3. OPERATIONS RESEARCH MODEL (OR-TOOLS)
# ==========================================
@st.cache_data
def solve_timetable(sections_df, init_rooms, red_rooms, red_week, max_daily):
    model = cp_model.CpModel()
    
    num_weeks = 10
    num_days = 7  # 0: Mon, ..., 6: Sun
    num_slots = 6 # 1.5 hr slots (Continuous setup until evening)
    
    num_sections = len(sections_df)
    
    # Generate Boolean Variables
    x = {}
    for c in range(num_sections):
        for w in range(num_weeks):
            for d in range(num_days):
                for s in range(num_slots):
                    x[c, w, d, s] = model.NewBoolVar(f'x_{c}_{w}_{d}_{s}')
                    
    # Constraint 1: Exactly 20 sessions per section
    for c in range(num_sections):
        model.Add(sum(x[c, w, d, s] for w in range(num_weeks) for d in range(num_days) for s in range(num_slots)) == 20)
        
    # Constraint 2: Dynamic Classroom Capacity Limit per slot
    for w in range(num_weeks):
        current_capacity = init_rooms if (w + 1) < red_week else red_rooms
        for d in range(num_days):
            for s in range(num_slots):
                model.Add(sum(x[c, w, d, s] for c in range(num_sections)) <= current_capacity)
                
    # Constraint 3: Faculty Overlap
    faculty_groups = sections_df.groupby('Faculty').groups
    for faculty, indices in faculty_groups.items():
        for w in range(num_weeks):
            for d in range(num_days):
                for s in range(num_slots):
                    model.Add(sum(x[c, w, d, s] for c in indices) <= 1)
                    
    # Constraint 4: NEW - Student Overlapping (Conflict Matrix)
    # If two sections share even 1 student, they cannot be scheduled at the exact same time
    for c1 in range(num_sections):
        for c2 in range(c1 + 1, num_sections):
            set1 = sections_df.iloc[c1]['Student_IDs']
            set2 = sections_df.iloc[c2]['Student_IDs']
            
            if not set1.isdisjoint(set2): # If they share students -> Conflict exists
                for w in range(num_weeks):
                    for d in range(num_days):
                        for s in range(num_slots):
                            # Both cannot be true (1) at the same time
                            model.AddImplication(x[c1, w, d, s], x[c2, w, d, s].Not())
                    
    # Constraint 5: Max Daily Sessions per Section
    for c in range(num_sections):
        for w in range(num_weeks):
            for d in range(num_days):
                model.Add(sum(x[c, w, d, s] for s in range(num_slots)) <= max_daily)
                
    # Constraint 6: Sunday specific timings (Sundays until 15:45 -> Block Slot 5 & 6)
    for c in range(num_sections):
        for w in range(num_weeks):
            model.Add(x[c, w, 6, 4] == 0) # Slot 5 (Evening)
            model.Add(x[c, w, 6, 5] == 0) # Slot 6 (Late Evening)
            
    # Soft Objective: Minimize weekend classes (Sat/Sun = days 5 and 6)
    weekend_penalty = sum(x[c, w, d, s] for c in range(num_sections) for w in range(num_weeks) for d in [5, 6] for s in range(num_slots))
    model.Minimize(weekend_penalty)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 45.0 # Increased timeout due to heavy student constraints
    
    status = solver.Solve(model)
        
    schedule = []
    unscheduled = 0
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Reconstruct Schedule & Assign specific Classrooms
        slot_labels = [
            "09:00 - 10:30 (Slot 1)",
            "10:45 - 12:15 (Slot 2)",
            "12:30 - 14:00 (Slot 3)",
            "14:15 - 15:45 (Slot 4)",
            "16:00 - 17:30 (Slot 5)",
            "17:45 - 19:15 (Slot 6)"
        ]
        
        for w in range(num_weeks):
            for d in range(num_days):
                for s in range(num_slots):
                    active_sections = [c for c in range(num_sections) if solver.Value(x[c, w, d, s]) == 1]
                    
                    for i, c in enumerate(active_sections):
                        row = sections_df.iloc[c]
                        schedule.append({
                            "Course": row['Course'],
                            "Section": row['Section'],
                            "Faculty": row['Faculty'],
                            "Week": w + 1,
                            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                            "Slot": slot_labels[s],
                            "Slot_Idx": s,
                            "Room": f"CR-{i+1}"
                        })
    else:
        unscheduled = num_sections * 20

    return pd.DataFrame(schedule), unscheduled, status

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("MBA Timetable Optimization Dashboard")
st.markdown("This dashboard uses **Google OR-Tools** to resolve scheduling conflicts. It explicitly enforces a **Student Conflict Matrix**, dynamically calculating shared enrolments across uploaded files to ensure no student is ever double-booked.")

# File Uploader - Now accepts a single Excel file
st.markdown("### 📂 Upload IIM Ranchi Course Data (Master Excel File)")
uploaded_file = st.file_uploader("Upload the Master Course Excel file (.xlsx) containing all sheets", type=['xlsx', 'xls'])

if not uploaded_file:
    st.info("Please upload your Master Course Excel file to generate the timetable.")
    st.stop()

# Process Data & Run Optimization
with st.spinner("Processing Excel Sheets & Solving Network Constraints..."):
    course_data, sections_data = process_uploaded_excel(uploaded_file, max_capacity)
    
    # Generate conflict metrics for UI
    total_students = len(set(x for l in course_data['Students_List'] for x in l))
    
    schedule_df, unscheduled_count, solver_status = solve_timetable(
        sections_data, initial_rooms, reduced_rooms, reduction_week, max_daily_sessions_student
    )

# ==========================================
# 5. DEPENDENT VARIABLES & KPIs
# ==========================================
total_capacity_slots = 0
for w in range(1, 11):
    cap = initial_rooms if w < reduction_week else reduced_rooms
    total_capacity_slots += (cap * 39) # 6 slots * 6 days + 3 slots on Sunday = 39 slots per week per room

total_sections = len(sections_data)
total_sessions_scheduled = len(schedule_df)
avg_utilization = (total_sessions_scheduled / total_capacity_slots * 100) if total_capacity_slots > 0 else 0

st.markdown("### 🏆 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Unique Students Processed", total_students)
col2.metric("Total Active Sections", total_sections, delta=f"{total_sections - len(course_data)} split sections", delta_color="inverse")
col3.metric("Overall Room Utilization", f"{avg_utilization:.1f}%")

if solver_status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    col4.metric("Unscheduled/Conflicts", f"{unscheduled_count} Sessions", delta="Feasible Schedule ✅", delta_color="normal")
else:
    col4.metric("Unscheduled/Conflicts", "INFEASIBLE", delta="Try relaxing constraints ❌", delta_color="inverse")
    st.error("The solver could not find a feasible schedule. With the strict 'No Student Overlap' constraint, you likely need to increase Maximum Classrooms or reduce the Max Daily limit.")
    st.stop()

st.markdown("---")

# ==========================================
# 6. VISUAL INSIGHTS (PLOTLY)
# ==========================================
st.markdown("### 📊 Visual Analytics & Insights")

c1, c2 = st.columns(2)

# Insight 1: Bar chart of Course Enrollments & Splits
with c1:
    fig1 = px.bar(course_data, x="Course", y="Total_Students", color="Total_Students", 
                  title="Course Enrollments (Dashed line triggers Section Splits)",
                  color_continuous_scale=px.colors.sequential.Teal)
    fig1.add_hline(y=max_capacity, line_dash="dash", annotation_text="Max Capacity Limit", line_color="red")
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

# Insight 2: Line chart showing drop in classroom availability vs usage
with c2:
    weekly_usage = schedule_df.groupby('Week').size().reset_index(name='Used Sessions')
    
    capacity_data = []
    for w in range(1, 11):
        cap = initial_rooms if w < reduction_week else reduced_rooms
        capacity_data.append({"Week": w, "Capacity Limit": cap * 39})
    cap_df = pd.DataFrame(capacity_data)
    
    usage_vs_cap = pd.merge(weekly_usage, cap_df, on="Week", how="right").fillna(0)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=usage_vs_cap['Week'], y=usage_vs_cap['Used Sessions'], mode='lines+markers', name='Actual Usage', line=dict(color='#1f77b4', width=3)))
    fig2.add_trace(go.Scatter(x=usage_vs_cap['Week'], y=usage_vs_cap['Capacity Limit'], mode='lines', name='Total Capacity', line=dict(color='#ff7f0e', width=2, dash='dash')))
    fig2.update_layout(title="Classroom Utilization vs Available Capacity Drop", xaxis_title="Week", yaxis_title="Total Sessions", plot_bgcolor="rgba(0,0,0,0)")
    fig2.add_vline(x=reduction_week - 0.5, line_width=1, line_dash="dash", line_color="red", annotation_text="Capacity Reduction")
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

# Insight 3: Histogram of Faculty Workload
with c3:
    faculty_workload = schedule_df.groupby('Faculty').size().reset_index(name='Total Sessions')
    fig3 = px.bar(faculty_workload.sort_values('Total Sessions', ascending=False), 
                  x="Faculty", y="Total Sessions", 
                  title="Faculty Workload Distribution (Over 10 Weeks)",
                  color="Total Sessions", color_continuous_scale="Blues")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

# Insight 4: Pie chart of Morning vs Evening sessions
with c4:
    # Updated categorization based on new timings
    def categorize_slot(idx):
        if idx in [0, 1]: return 'Morning (09:00 - 12:15)'
        elif idx == 2: return 'Afternoon (12:30 - 14:00)'
        else: return 'Evening (14:15 - 19:15)'
        
    schedule_df['Shift Category'] = schedule_df['Slot_Idx'].apply(categorize_slot)
    time_usage = schedule_df.groupby('Shift Category').size().reset_index(name='Count')
    
    fig4 = px.pie(time_usage, names='Shift Category', values='Count', 
                  title="Distribution of Classes by Time of Day", hole=0.4,
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# 7. MASTER TIMETABLE DATA
# ==========================================
st.markdown("### 📋 Master Timetable Output")
st.caption("Complete mapping of Courses, Sections, Faculty, and Time/Room Allocations (Optimized constraint satisfaction).")

# Drop the helper index for clean presentation
display_df = schedule_df.drop(columns=['Slot_Idx', 'Shift Category']).sort_values(by=["Week", "Day", "Slot", "Room"])
st.dataframe(display_df, use_container_width=True)
