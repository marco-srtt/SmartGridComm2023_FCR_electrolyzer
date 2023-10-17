
# FCR-D Down DETERMINISTIC MODEL ELECTROLYZER (2 binaries for states)
# The objective of thi model is to investigate adn quantify the additional value
# for an electrolyzer asset for the provision of ancillary services in Denmark

### PACKAGE IMPORT
#using JuMP, XLSX, CSV, DataFrames,  Gurobi, Plots, Dates, Statistics
using JuMP,  XLSX, CSV, DataFrames,  Gurobi, Dates, Statistics
# pwd() command to check directory
# cd("C:\\Users\\Marco Saretta\\Documents\\GitHub\\videnbro_ancillary_services")  to set and change directory

### MODEL DEFINITION
det_FCR_multi = Model(Gurobi.Optimizer)  # Create a model with Gurobi Optimizer

### IMPORT DATA
raw_data = CSV.read("input_deterministic.csv", DataFrame)
raw_data.date = Date.(raw_data.date, "m/d/y H:M")   # Parsing dates as datetime format

# Slice years

# Year selection
select_year = 2022

# Dataset slicing
date_rows_price = findall(d -> year(d) == select_year, raw_data.date)

df = raw_data[date_rows_price, :]

### PARAMETERS

# Alkaline electrolyzer
E = 10        # Electrolyzer capacity
P_min = 0.16      # % min. power, related to total load
P_max = 1         # % max. power, related to total load
P_sb = 0.05      # % Standby power, related to total load
eta_sys = 0.60      # % of system efficiency      https://9441822.fs1.hubspotusercontent-na1.net/hubfs/9441822/Brochures/Technical%20specs/Technical%20Overview_2022-09-22.pdf
LHV_h2_kWh = 33.33     # kWh/kg of hydrogen
HHV_h2_kWh = 39.41     # kWh/kg of hydrogen
LHV_h2_MWh = LHV_h2_kWh / 1000      # kWh/kg of hydrogen
HHV_h2_MWh = HHV_h2_kWh / 1000      # kWh/kg of hydrogen
cold_start_cost = 100   # Cold start up cost [€/MW]
compresor_factor = 0.05 * LHV_h2_MWh # Power to compress [MWh/kg]

# Piecewise 
# 4 segment
B_0 = [-4.951477, -2.151771, 3.811083, 13.248698, 22.821486]
B_1 = [21.941757, 20.541904, 18.428714, 16.541191, 15.264820]
p_segment = [0.16, 0.2, 0.28374864, 0.5, 0.75, 1]

# Prices
lambda_h2 = 2               # Price for Hydrogen sales

FCR_factor = 1
lambda_spot = df[:, 2]      # Spot price [€/MWh]
lambda_FCRN = FCR_factor*df[:, 3]      # FCRN reserve bid price   [€/MW]
lambda_FCRD_up = FCR_factor*df[:, 4]   # FCRD up reserve bid price   [€/MW]
lambda_FCRD_down = FCR_factor*df[:, 5] # FCRD down reserve bid price   [€/MW]


# Tariffs
conversion_fact = 10 / 7.45
tariff_TSO_c_ore = 11.229
tariff_TSO_p_ore = 0.416
tariff_DSO_ore = Statistics.mean([2.13, 3.98, 5.88])

tariff_TSO_c_eur = tariff_TSO_c_ore * conversion_fact
tariff_TSO_p_eur = tariff_TSO_p_ore * conversion_fact
tariff_DSO_eur = tariff_DSO_ore * conversion_fact

# Demand
demand_factor = 0.3      # Lower bound of percentage of weekly demand to be delivered, evaluated on total weekly H2 demand
tank_discharge = 0.2       # Tank size, percentage evaluated on total weekly H2 demand
tank_size = 2

days_df = size(df)[1] ÷ 24            # Numebr of days in DataFrame
weeks_df = size(df)[1] ÷ (7 * 24)     # Numebr of weeks in DataFrame
months_df = size(df)[1] ÷ (30 * 24)   # Numebr of months in DataFrame
periods_daily = collect(1:days_df)        # Creates array uniformely spaced, size of days_df
periods_weeks = collect(1:weeks_df)       # Creates array uniformely spaced, size of weeks_df
periods_month = collect(1:months_df)      # Creates array uniformely spaced, size of motnhs_df

### SETS
periods = collect(1:size(df)[1])        # Creates array uniformely spaced, size of df
segments = collect(1:length(p_segment)-1)

### VARIABLES

# Electrolyzer variables
@variable(det_FCR_multi, x[t in periods] >= 0)           # Load dispatched electrolyer         [MW]
@variable(det_FCR_multi, x_elyzer[t in periods] >= 0)    # Load dispatched electrolyer         [MW]
@variable(det_FCR_multi, r_fcrd_up[t in periods] >= 0)   # Reserve  power for the services     [MW]
@variable(det_FCR_multi, r_fcrd_down[t in periods] >= 0) # Reserve  power for the services     [MW]
@variable(det_FCR_multi, r_fcrn[t in periods] >= 0)      # Reserve  power for the services     [MW]
@variable(det_FCR_multi, h2[t in periods] >= 0)          # Hydrogen production                 [kg]

# Binary variables
@variable(det_FCR_multi, z_on[t in periods], Bin)    # Online state electrolyzer
@variable(det_FCR_multi, z_sb[t in periods], Bin)    # Standby state electrolyzer
#@variable(det_FCR_multi, z_off[t in periods], Bin)   # Offline state electrolyzer
@variable(det_FCR_multi, c[t in periods], Bin)       # Cold start up cost to be accounted for

# Piecewise linearization
@variable(det_FCR_multi, z_s[t in periods, s in segments], Bin)    # Activation for segment of the curve
@variable(det_FCR_multi, p_s[t in periods, s in segments] >= 0)    # Activation for segment of the curve

# Demand variables
@variable(det_FCR_multi, d[t in periods] >= 0)     # Hydrogen demand to be supplied

# Storage variables
@variable(det_FCR_multi, h2_tank[t in periods] >= 0) # Hydrogen tank

# Compressor variables
@variable(det_FCR_multi, x_compr[t in periods] >= 0) # Hydrogen compressor power

# CONSTRAINTS

# States
@constraint(det_FCR_multi, mutual_excl_states[t in periods], z_on[t] + z_sb[t] <= 1)        # Ensures mutual exclusion of the states

# Piecewise
@constraint(det_FCR_multi, mutual_excl_segment[t in periods], sum(z_s[t, s] for s in segments) == z_on[t])        # Ensures mutual exclusion of the states
@constraint(det_FCR_multi, power_segmentation[t in periods], sum(p_s[t, s] for s in segments) == x_elyzer[t] + P_sb * E * z_sb[t])
@constraint(det_FCR_multi, lower_segment_bound[t in periods, s in segments], p_s[t, s] >= z_s[t, s] * p_segment[s] * E)
@constraint(det_FCR_multi, upper_segment_bound[t in periods, s in segments], p_s[t, s] <= z_s[t, s] * p_segment[s+1] * E)

# Power balance constraint 
@constraint(det_FCR_multi, power_bal[t in periods], x_elyzer[t] + x_compr[t] == x[t])  # Power bounded to minimum setpoint

# Power
@constraint(det_FCR_multi, power_bal_low[t in periods], P_min * E * z_on[t] + P_sb * E * z_sb[t] <= x_elyzer[t])  # Power bounded to minimum setpoint
@constraint(det_FCR_multi, power_bal_up[t in periods], P_max * E * z_on[t] + P_sb * E * z_sb[t] >= x_elyzer[t])  # Power bounded to maximum setpoint

# Cold startup cost
@constraint(det_FCR_multi, cold_start_t0[t=1], c[t] == 0)
@constraint(det_FCR_multi, cold_start[t in periods; t > 1], c[t] >= (z_on[t]-z_on[t-1]) + (z_sb[t] - z_sb[t-1]))

# Hydrogen
#@constraint(det_FCR_multi, hydrogen_low[t in periods], h2[t] == (eta_sys * (x_elyzer[t] - (P_sb) * z_sb[t] * E)) / LHV_h2_MWh)    # Hydrogen production constraint
@constraint(det_FCR_multi, hydrogen_low[t in periods], h2[t] == sum(B_0[s] * z_s[t, s] + B_1[s] * p_s[t, s] for s in segments))    # Hydrogen production constraint
@constraint(det_FCR_multi, hydrogen_compr[t in periods], x_compr[t] == h2[t] * compresor_factor)


# Reserve contraint
@constraint(det_FCR_multi, reserve_FCR_up[t in periods], x_elyzer[t] - r_fcrd_up[t] - r_fcrn[t] >= P_min * E * z_on[t] + P_sb * E * z_sb[t])      # Lower bound of the reserve power
@constraint(det_FCR_multi, reserve_FCR_down[t in periods], x_elyzer[t] + r_fcrd_down[t] + r_fcrn[t] <= P_max * E * z_on[t] + P_sb * E * z_sb[t])  # Upper bound of the reserve power

# Minimum demand 
for i in periods_weeks
  local test = periods[((i-1)*24*7+1):(i*(24*7))]
  @constraint(det_FCR_multi, sum(d[t] for t in test) >= ((eta_sys * length(test) * E) / LHV_h2_MWh) * demand_factor)
end

@constraint(det_FCR_multi, [t=1], h2_tank[t] == h2[t] - d[t])
@constraint(det_FCR_multi, [t in periods; t > 1], h2_tank[t] == h2_tank[t-1] + h2[t] - d[t])
@constraint(det_FCR_multi, [t in periods; t > 1], h2_tank[t] - h2_tank[t-1] <= ((eta_sys * (24 * 7) * E) / LHV_h2_MWh) * tank_size * tank_discharge)

@constraint(det_FCR_multi, [t in periods], h2_tank[t] <= ((eta_sys * (24 * 7) * E) / LHV_h2_MWh) * tank_size)


# Declare the objective function for the deterministic model
@objective(det_FCR_multi, Max, sum(r_fcrn[t] * lambda_FCRN[t] + r_fcrd_up[t] * lambda_FCRD_up[t] + r_fcrd_down[t] * lambda_FCRD_down[t] + d[t] * lambda_h2 - x[t] * (lambda_spot[t] + tariff_TSO_c_eur + tariff_DSO_eur) - c[t] * cold_start_cost * E for t in periods))

# Print the cmoplete model
#print(det_FCR_multi)

optimize!(det_FCR_multi)

println("  ")
println("The objective value of the model (Expected profit) is ", objective_value(det_FCR_multi))
println("  ")

### STORE RESULTS

# Solution vectors initialization
x_SOL = zeros(length(periods))
x_elyzer_SOL = zeros(length(periods))
x_compr_SOL = zeros(length(periods))
r_fcrn_SOL = zeros(length(periods))
r_fcrd_up_SOL = zeros(length(periods))
r_fcrd_down_SOL = zeros(length(periods))
h2_SOL = zeros(length(periods))
z_on_SOL = zeros(length(periods))
z_sb_SOL = zeros(length(periods))
z_off_SOL = zeros(length(periods))
c_SOL = zeros(length(periods))
profit_SOL = zeros(length(periods))

# Demand vector initialization
h2_tank_SOL = zeros(length(periods))
demand_SOL = zeros(length(periods))

# Profit decomposition vectors initialization
R_FCRN_SOL = zeros(length(periods))
R_FCRD_up_SOL = zeros(length(periods))
R_FCRD_down_SOL = zeros(length(periods))
R_H_SOL = zeros(length(periods))
C_baseline_SOL = zeros(length(periods))
C_tariff_SOL = zeros(length(periods))
C_start_SOL = zeros(length(periods))

# Piecewise
z_s_SOL = zeros(length(periods), length(segments))
p_s_SOL = zeros(length(periods), length(segments))


for t in periods
  # Profit anaylsis
  x_SOL[t] = value(x[t])
  r_fcrd_up_SOL[t] = value(r_fcrd_up[t])
  r_fcrd_down_SOL[t] = value(r_fcrd_down[t])
  r_fcrn_SOL[t] = value(r_fcrn[t])
  h2_SOL[t] = value(h2[t])
  z_on_SOL[t] = value(z_on[t])
  z_sb_SOL[t] = value(z_sb[t])
  #z_off_SOL[t] = value(z_off[t])
  c_SOL[t] = value(c[t])
  profit_SOL[t] = value(r_fcrn[t] * lambda_FCRN[t] + r_fcrd_up[t] * lambda_FCRD_up[t] + r_fcrd_down[t] * lambda_FCRD_down[t] + h2[t] * lambda_h2 - x[t] * (lambda_spot[t] + tariff_TSO_c_eur + tariff_DSO_eur) - c[t] * cold_start_cost * E)

  # Demand analysis
  h2_tank_SOL[t] = value(h2_tank[t])
  demand_SOL[t] = value(d[t])
  x_elyzer_SOL[t] = value(x_elyzer[t])
  x_compr_SOL[t] = value(x_compr[t])

  # Revenue analysis
  R_FCRN_SOL[t] = value(r_fcrn[t] * lambda_FCRN[t])
  R_FCRD_up_SOL[t] = value(r_fcrd_up[t] * lambda_FCRD_up[t])
  R_FCRD_down_SOL[t] = value(r_fcrd_down[t] * lambda_FCRD_down[t])
  R_H_SOL[t] = value(h2[t] * lambda_h2)
  C_baseline_SOL[t] = value(-x[t] * lambda_spot[t])
  C_tariff_SOL[t] = value(-x[t] * (tariff_TSO_c_eur + tariff_DSO_eur))
  C_start_SOL[t] = value(-c[t] * cold_start_cost * E)

  # Piecewise 
  for s in segments
    z_s_SOL[t, s] = value(z_s[t, s])
    p_s_SOL[t, s] = value(p_s[t, s])
  end
end

### VISUALIZE RESULTS

#bar(periods, profit_SOL) # Barplot of results
#p1 = plot(periods, x_SOL)
#println(describe(profit_SOL))

### EXPORT RESULTS

# Create a unique matrix with all solutions values

SOL_MATRIX = hcat(x_SOL, x_elyzer_SOL, x_compr_SOL, r_fcrd_up_SOL, r_fcrd_down_SOL, r_fcrn_SOL, h2_SOL, z_on_SOL, z_sb_SOL, z_off_SOL, c_SOL, profit_SOL, lambda_FCRN, lambda_FCRD_down, lambda_FCRD_up, lambda_spot)
PROFIT_ANALYSIS = hcat(R_FCRN_SOL, R_FCRD_up_SOL, R_FCRD_down_SOL, R_H_SOL, C_baseline_SOL, C_tariff_SOL, C_start_SOL, profit_SOL)
DEMAND_ANALYSIS = hcat(h2_SOL, demand_SOL, h2_tank_SOL,)
PIECEWISE_ANALYSIS = hcat(z_s_SOL, p_s_SOL)

# Create labels for the CSV columns
SOL_label = ["Power_elyzer", "x_elyzer_SOL", "x_compr_SOL", "r_FCRD_up", "r_FCRD_down", "r_FCRN", "h2_SOL", "z_on_SOL", "z_sb_SOL", "z_off_SOL", "c_SOL", "profit_SOL", "lambda_FCRN", "lambda_FCRD_down", "lambda_FCRD_up", "lambda_spot"]
PROFIT_ANALYSIS_label = ["revenue_FCRN", "revenue_FCRD_up", "revenue_FCRD_down", "revenue_H2", "C_baseline", "C_tariff", "C_start", "profit"]
DEMAND_ANALYSIS_label = ["h2", "demand", "tank"]
PIECEWISE_ANALYSIS_label = hcat("z_s_SOL", "p_s_SOL")

# Create DataFrame
SOL_df = DataFrame(Tables.table(SOL_MATRIX))
PROFIT_ANALYSIS_df = DataFrame(Tables.table(PROFIT_ANALYSIS))
DEMAND_ANALYSIS_df = DataFrame(Tables.table(DEMAND_ANALYSIS))
PIECEWISE_ANALYSIS_df = DataFrame(Tables.table(PIECEWISE_ANALYSIS))

# Rename individual columns
rename!(SOL_df, SOL_label)
rename!(PROFIT_ANALYSIS_df, PROFIT_ANALYSIS_label)
rename!(DEMAND_ANALYSIS_df, DEMAND_ANALYSIS_label)
#rename!(PIECEWISE_ANALYSIS, PIECEWISE_ANALYSIS_label)

# Remove prvious document
rm("results\\FCR_multi\\det_FCR_multi_flex_piece.xlsx", force=true)

# Write results
#XLSX.writetable("results\\FCR_multi\\det_FCR_multi_flex_piece.xlsx", system_operation=(collect(eachcol(SOL_df)), names(SOL_df)), profit_breakdown=(collect(eachcol(PROFIT_ANALYSIS_df)), names(PROFIT_ANALYSIS_df)), demand_analysis=(collect(eachcol(DEMAND_ANALYSIS_df)), names(DEMAND_ANALYSIS_df)), piecewise=(collect(eachcol(PIECEWISE_ANALYSIS_df)), names(PIECEWISE_ANALYSIS_df)))
#CSV.write("results/FCR_multi/det_FCR_multi_flex_piece.csv", (
    #system_operation = (collect(eachcol(SOL_df)), names(SOL_df)),
    #profit_breakdown = (collect(eachcol(PROFIT_ANALYSIS_df)), names(PROFIT_ANALYSIS_df)),
    #demand_analysis = (collect(eachcol(DEMAND_ANALYSIS_df)), names(DEMAND_ANALYSIS_df)),
    #piecewise = (collect(eachcol(PIECEWISE_ANALYSIS_df)), names(PIECEWISE_ANALYSIS_df))
#))
CSV.write("results/FCR_multi/det_FCR_multi_flex_piece_2bin_solution.csv", SOL_df)
CSV.write("results/FCR_multi/det_FCR_multi_flex_piece_2bin_profit.csv", PROFIT_ANALYSIS_df)
CSV.write("results/FCR_multi/det_FCR_multi_flex_piece_2bin_demand.csv", DEMAND_ANALYSIS_df)
CSV.write("results/FCR_multi/det_FCR_multi_flex_piece_2bin_piecewise.csv", PIECEWISE_ANALYSIS_df)

# Check solution time
solution_summary(det_FCR_multi)
