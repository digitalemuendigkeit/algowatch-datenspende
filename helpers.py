import datetime

# wrapper method to apply timestamp method to column
def get_timestamp(x):
   return x.timestamp()

def get_country(x):
   return x.split()[0]