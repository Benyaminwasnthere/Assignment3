import subprocess
for i in range(5):
    for n in range(1,3):
        for l in range(1, 3):
            if l%2 ==0:
                variance="H"
            else:
                variance = "L"
            childproc = subprocess.Popen(f'C:\\Users\Ben\PycharmProjects\Assignment3\\venv\\Scripts\\python.exe simulate.py --plan controls/controls_{i}_{n}.npy --map maps/landmark_{i}.npy --sensing readings/readings_{i}_{n}_{variance}.npy')
    op, oe = childproc.communicate()
    print(op)
