from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = rf.initial_control()

lb, ub = IPOptUtils.position_constraints(config) 
minimize(rf, bounds = [lb, ub], method = "SLSQP", options = {"maxiter": 200})
