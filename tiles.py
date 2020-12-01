'''Two parts of a floor need tiling. One part is 9 tiles wide by 7 tiles long, the other is 5 tiles wide by 7 tiles long. Tiles come in packages of 6.

How many tiles are needed?
You buy 17 packages of tiles containing 6 tiles each. How many tiles will be left over?'''

tiles_needed = 9*7+5*7
print(tiles_needed)
tiles_left = 17*6-tiles_needed
print(tiles_left)
