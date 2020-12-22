############################################
#
# Module for Method of Manufactured Solutions (MMS)
#
############################################


from sympy import symbols, sin, diff, lambdify, pi
from variables import *


# Manufactured Solutions
y = symbols('y') # Position
s = symbols('s') # Time


# MMS settings
rho_0 = 1
rho_x = 0.1
ax_rho = 10

u_0 = 1
u_x = 0.5
ax_u = 6

p_0 = 1
p_x = -0.5
ax_p = 2


# Symbolic expressions
rho_expr = rho_0 + rho_x*sin(ax_rho*pi*(y-1.5*s)/L)
u_expr = u_0 + u_x*sin(ax_u*pi*(y-1.5*s)/L)
p_expr = p_0 + p_x*sin(ax_p*pi*(y-1.5*s)/L)
e_expr = p_expr/(gamma-1)/rho_expr + 0.5*u_expr**2

f_rho_expr = diff(rho_expr, s) + diff(rho_expr*u_expr, y)
f_u_expr = diff(rho_expr*u_expr, s) \
			+ diff(rho_expr*u_expr**2+p_expr-1/Re*diff(u_expr, y), y)
f_e_expr = diff(rho_expr*e_expr, s) \
			+ diff(u_expr*(gamma*p_expr/(gamma-1)+0.5*rho_expr*u_expr**2) \
			- u_expr/Re*diff(u_expr,y) - gamma/(gamma-1)/Re/Pr*diff(p_expr/rho_expr, y),y)


# Callable functions
rho = lambdify([y,s], rho_expr)
u = lambdify([y,s], u_expr)
p = lambdify([y,s], p_expr)
e = lambdify([y,s], e_expr)

f_rho = lambdify([y,s], f_rho_expr)
f_u = lambdify([y,s], f_u_expr)
f_e = lambdify([y,s], f_e_expr)

