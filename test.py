import tee

tee.startTee('./newTestFolder')
print('\nThis should output to both')
print('What fun.')
tee.endTee()