import pykep
print(pykep.__version__)
from pykep import AU, DAY2SEC, MU_SUN

print("1 AU in meters:", AU)
print("1 day in seconds:", DAY2SEC)
print("Sun gravitational parameter:", MU_SUN)