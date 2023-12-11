import subprocess
num=50
for i in range(5):
    for n in range(1,3):
        for l in range(1, 3):
            if l%2 ==0:
                variance="H"
            else:
                variance = "L"
            childproc = subprocess.Popen(f'C:\\Users\Ben\PycharmProjects\Assignment3\\venv\\Scripts\\python.exe particle_filter_kidnapped.py --map maps/landmark_{i}.npy --sensing readings/readings_{i}_{n}_{variance}.npy --num_particles {num}')
            op, oe = childproc.communicate()
    print(op)
num=5000
for i in range(5):
    for n in range(1,3):
        for l in range(1, 3):
            if l%2 ==0:
                variance="H"
            else:
                variance = "L"
            childproc = subprocess.Popen(f'C:\\Users\Ben\PycharmProjects\Assignment3\\venv\\Scripts\\python.exe particle_filter_kidnapped.py --map maps/landmark_{i}.npy --sensing readings/readings_{i}_{n}_{variance}.npy --num_particles {num}')
            op, oe = childproc.communicate()
    print(op)
num=50
for i in range(5):
    for n in range(1,3):
        for l in range(1, 3):
            if l%2 ==0:
                variance="H"
            else:
                variance = "L"
            childproc = subprocess.Popen(f'C:\\Users\Ben\PycharmProjects\Assignment3\\venv\\Scripts\\python.exe particle_filter.py --map maps/landmark_{i}.npy --sensing readings/readings_{i}_{n}_{variance}.npy --num_particles {num}')
            op, oe = childproc.communicate()
    print(op)
num=5000
for i in range(5):
    for n in range(1,3):
        for l in range(1, 3):
            if l%2 ==0:
                variance="H"
            else:
                variance = "L"
            childproc = subprocess.Popen(f'C:\\Users\Ben\PycharmProjects\Assignment3\\venv\\Scripts\\python.exe particle_filter.py --map maps/landmark_{i}.npy --sensing readings/readings_{i}_{n}_{variance}.npy --num_particles {num}')
            op, oe = childproc.communicate()
    print(op)