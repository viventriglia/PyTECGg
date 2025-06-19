import time

from georinex import load

start = time.time()
obs = load("rinex/v2/obs/cgtc0920.14o")
end = time.time()

print(f"Tempo di caricamento: {end - start:.2f} secondi")
# print(obs.info())

start = time.time()
obs = load("rinex/v3/obs/ASIR00ITA_R_20242810000_01D_30S_MO.crx.gz", use="G")
end = time.time()

print(f"Tempo di caricamento: {end - start:.2f} secondi")
# print(obs.info())
