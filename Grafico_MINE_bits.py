
import numpy as np
import matplotlib.pyplot as plt

layer = ['Layer1', 'Layer2', 'Layer3', 'Output Layer']

# Valores originales estimados en nats
output_gender = [0.3276190647277339, 0.2458402910109224,
                 0.13524605515252414, 0.06652485871494844]

output_race = [0.09313273247620411, 0.041336925844822466,
               0.013126089825728458, 0.008247186066502512]

output_age = [0.3124266857730931, 0.17756651984206562,
              0.13519192093597918, 0.10290605069041767]

output_money = [0.23536524202289252, 0.24309817534582367,
                0.24737633462866832, 0.23254723858152485]

# Conversión de nats a bits
output_gender = np.array(output_gender) / np.log(2)
output_race = np.array(output_race) / np.log(2)
output_age = np.array(output_age) / np.log(2)
output_money = np.array(output_money) / np.log(2)

# Gráfico
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(layer, output_gender, marker='o', label="Gender")
ax.plot(layer, output_race, marker='s', label="Race")
ax.plot(layer, output_age, marker='^', label="Age")
ax.plot(layer, output_money, marker='d', label="Money")

ax.set_title("Mutual Information Estimated by MINE")
ax.set_xlabel("Layer")
ax.set_ylabel("Mutual Information (bits)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
