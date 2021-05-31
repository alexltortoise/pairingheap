'''
Pairing Heap
(Last Edit: 20190827)

# # for details:
# # https://en.wikipedia.org/wiki/Pairing_heap
# # https://datastructures.maximal.io/heaps/pairing-heap/
# # https://gist.github.com/kachayev/5990757
# # https://github.com/mikolalysenko/pairing-heap/blob/master/heap.js

Here we implement Pairing Heap by Nested List. 
The 1st item of every list and sublist is always the MIN (in min heap) or MAX (in max heap).

eg. of min heap: [1, 34, [99, [108,278]], 2, [6], 42,....]
eg. of max heap: [999, [32,8,23], 65, [44, [4], [30,20]], ....]

'''


__all__ = ['minheapify','minmerge','minpop','flatten','size','depth','maxheapify','maxmerge','maxpop','isMinpheap','isMaxpheap']


def minheapify(arr):
	'''
	Create a MIN pairing heap by merging elements in the list 1 by 1. 
	Out-of-place. O(n) Time, O(n) Space.
	
	eg.	x = [89324,21,6,5,10,2,98];  minheapify(x)
	steps:
	[21,89324]
	[6,[21,89324]]
	[5,[6,[21,89324]]]
	[5,[6,[21,89324]],10]
	[2,[5,[6,[21,89324]],10]]
	[2,[5,[6,[21,89324]],10],98]
	'''
	n = len(arr)
	if n <= 1:
		return arr

	prev = [arr[0]]
	# # for i in range(1, n):
	for next in arr[1:]:
		# # next = arr[i]
		if prev[0] < next:
			prev.append(next)
		else:
			prev = [next,prev]		
	return prev


	
def maxheapify(arr):
	'''
	Same as minheapify(), but create a MAX pairing heap.
	'''
	n = len(arr)
	if n <= 1:
		return arr

	prev = [arr[0]]
	for next in arr[1:]:
		if prev[0] > next:		# # the only change
			prev.append(next)
		else:
			prev = [next,prev]		
	return prev

	
	
def minmerge(*pheaps):
	'''
	Merge MIN pairing heaps / single elements into a single MIN pairing heap. 
	Out-of-place. O(1) time, O(n) space.
		
	# # Compare 1st MIN heaps with next from left to right, 
	# # make the heap with larger root a subtree, 
	# # right under the root of the other heap.
	
	# # This allows a merger done in O(1) time, much faster than other types of heap.
	# # And this is why pairing heap is popular in Dijkstra, where merger is frequent.
	'''
	
	subh0 = pheaps[0]
	
	# convert single item into list
	htype = type(subh0)
	if htype == int or htype == float or htype == tuple:
		subh0 = [pheaps[0]]
	
	root0 = subh0[0]
	
	for subh1 in pheaps[1:]:
		# convert single item into list
		htype = type(subh1)
		if htype == int or htype == float or htype == tuple:
			subh1 = [subh1]	
		
		root1 = subh1[0]

		# # print(subh0, root0, root1)
		
		if root1 < root0:
			subh0 = subh1 + [subh0]
			root0 = root1
		else:
			subh0 += [subh1]
			
	return subh0

	
		
def maxmerge(*pheaps):
	'''
	Merge MAX pairing heaps / single elements into a single MAX pairing heap. 
	Out-of-place. O(1) time, O(n) space.
	'''
	
	subh0 = pheaps[0]
	
	# convert single item into list
	htype = type(subh0)
	if htype == int or htype == float or htype == tuple:
		subh0 = [pheaps[0]]
	
	root0 = subh0[0]
	
	for subh1 in pheaps[1:]:
		# convert single item into list
		htype = type(subh1)
		if htype == int or htype == float or htype == tuple:
			subh1 = [subh1]	
		
		root1 = subh1[0]
		
		if root1 > root0:			# # the only change
			subh0 = subh1 + [subh0]
			root0 = root1
		else:
			subh0 += [subh1]
			
	return subh0
	

	
def minmerge1(*pheaps):
	'''
	Merging by pairing (2x2). 
	This version is much SLOWER than simply merge 1x1 !
	Maybe due to the small number of heaps input.
	'''
	n = len(pheaps)
	pheaps = list(pheaps)
	
	# convert every item into list
	i=0
	for h in pheaps:
		htype = type(h)
		if htype == int or htype == float or htype == tuple:
			pheaps[i] = [h]
		i+=1
		
	# pairing from left to right, 2 by 2. 
	while n>1:
		for i in range(0,n//2):
			(small,big)=(i,i+1) if pheaps[i][0]<pheaps[i+1][0] else (i+1,i)
			pheaps[small] += [pheaps[big]]
			del pheaps[big]
		n = (n+1)//2			

	return pheaps[0]	
	
	

def minpop(pheap):
	'''
	Pop MIN and re-heapify the rest.
	In-place. O(log n) time.
	# # Steps:
	# # 1. cached & pop the current root
	# # 2. convert every non-list item into list
	# # 3. pairing 2 by 2 from left to right
	# # (4. merge 1 by 1 from left to right)
	# # Pairing is much faster than merging(0.7s vs 246s, 23000 runs), 
	# # since it reduces sub-heaps to the least and speeds up searching the min root. 
	# # Final no. of sub-heap depends on the length of sub-heap 
	# # being unpacked (be the root heap) in the last round, where only 2 sub-heaps are being left.
	# # In pairing, lengths of 2 sub-heaps tend to be equal.
	# # While merging let 1 sub-heap grow to a big heap like a snowball, 
	# # which is likely to be unpacked finally, as it carries the min of larger part of the heap.
	'''
	
	root = pheap.pop(0)
	
	n = len(pheap)
	# # n=1 can be 1 long list, still many to do
	if n < 1:		
		return pheap
		
	# convert every item into list
	i=0
	for h in pheap:
		htype = type(h)
		if htype == int or htype == float or htype == tuple:
			pheap[i] = [h]
		i+=1
		
	# pairing from left to right, 2 by 2. 
	while n>1:
		for i in range(0,n//2):
			(small,big)=(i,i+1) if pheap[i][0]<pheap[i+1][0] else (i+1,i)
			pheap[small] += [pheap[big]]
			del pheap[big]
		n = (n+1)//2			
	
	# # # merge from left to right, 1 by 1.

	# # # merge: method 2
	# # n = len(pheap)
	# # for _ in range(n):
		# # (small,big)=(0,1) if pheap[0][0]<pheap[1][0] else (1,0)
		# # pheap[small] += [pheap[big]]
		# # del pheap[big]

	# # # merge: method 1
	# # prevroot = pheap[0][0]
	# # for h in pheap[1:]:
		# # nextroot = h[0]
		# # if nextroot < prevroot:
			# # pheap[1] += [pheap[0]] 
			# # del pheap[0]
			# # prevroot = nextroot
		# # else:
			# # pheap[0] += [pheap[1]]
			# # # pheap[0] += [h]
			# # del pheap[1]
	
	pheap += pheap.pop()		#remove the outermost bracket
	return root

	
	
def maxpop(pheap):
	'''
	Pop MAX and re-maxheapify the rest.
	In-place. O(log n) time.
	'''

	root = pheap.pop(0)
	
	n = len(pheap)
	# # n=1 can be 1 long list, still many to do
	if n < 1:		
		return pheap
		
	# convert every item into list
	i=0
	for h in pheap:
		htype = type(h)
		if htype == int or htype == float or htype == tuple:
			pheap[i] = [h]
		i+=1
		
	# pairing from left to right, 2 by 2. 
	while n>1:
		for i in range(0,n//2):
			(small,big)=(i,i+1) if pheap[i][0]<pheap[i+1][0] else (i+1,i)
			pheap[big] += [pheap[small]]		# # the only change
			del pheap[small]							# # the only change
		n = (n+1)//2		
		
	pheap += pheap.pop()		# remove the outermost bracket
	return root
	

	
def flatten(pheap):
	'''
	Return an iterator of flattened list. 
	Out-of-place. Iterative. O(n) time. O(n) space.
	
	eg. [for _ in flatten([2,[5,[6,[21,89324]],10],98])]  
	>>> [2,5,6,21,89324,10,98]
	'''
	for item in pheap:
		item = [item]
		while 1:
			try:
				child = item.pop(0)
				if type(child) == list:
					item = child + item
				else:
					yield child
			except IndexError:		# nothing to pop
				break
	

	
def flatten2(pheap):
	'''
	Return a copy of flattened list. 
	Out-of-place. Recrusive. O(n) time.
	'''
	flat = []
	for item in pheap:
		if type(item) == list:
			flat += flatten2(item)
		else:
			flat.append(item)
	return flat

def flatten3(pheap):
	'''
	Return an iterator of flattened list. 
	Out-of-place. Recrusive. O(n) time.
	'''
	for child in pheap:
		if type(child) == list:
			yield from flatten3(child)
		else:
			yield child

			
			
def size(pheap):
	'''
	Return total no. of element. 
	Iterative. O(log n) time. O(n) space for queue of sublists.
	'''
	n = len(pheap)
	queue1  = pheap
	queue2 = []
	while 1:
		
		for sub in queue1:
			if type(sub) == list:
				n += len(sub)-1
				queue2.append(sub)
		
		if not queue2:
			return n
			
		queue1 = queue2.pop()

		

def depth(pheap):
	'''
	Return no. of level of a pairing heap. 
	Start form level 0.
	Iterative. O(log n) time. O(n) space for list of sublists.
	'''
	queue1 = pheap
	queue2 = []
	level = 0
	while 1:
	
		for sub in queue1:
			if type(sub) == list:
				queue2 += sub
		
		if not queue2:
			return level

		queue1 = queue2
		queue2 = []
		level += 1


		
def isMinpheap(pheap):
	'''	
	Return a Boolean. True if the input is a MIN pairing heap, False otherwise.
	'''
	queue1 = pheap
	queue2 = []
	
	while 1:
		root = queue1[0]
		
		for child in queue1[1:]:
			if type(child) == list:
				if child[0] < root:
					return False
				else:
					queue2.append(child)
			elif child < root:
				return False
		
		if not queue2:
			return True
		
		queue1 = queue2.pop()
		

		
def isMaxpheap(pheap):
	''' 	 
	Return a Boolean. True if the input is a MAX pairing heap, False otherwise.
	'''
	queue1 = pheap
	queue2 = []
	
	while 1:
		root = queue1[0]
		
		for child in queue1[1:]:
			if type(child) == list:
				if child[0] > root:
					return False
				else:
					queue2.append(child)
			elif child > root:
				return False
		
		if not queue2:
			return True
		
		queue1 = queue2.pop()
		


		

if __name__== "__main__":
	# help(minheapify)
	x = [1,89324,21,6,5,10,2,98] *5
	#x = [9]
	a = [23,5,7,3,2,64,5,6,22,10,2,99,4,7,1,1] *1
	#x = [(89324,'dream'),(5,'reality'),(6,'hell'),(21,'heaven'),(10,'god'),(2,'die'),(98,'fly')]
	#a = [(23,'a'),(5,'r'),(7,'ez'),(3,'op'),(2,'f7'),(64,'banana'),(5,'qq'),(6,'shi'),(22,'go')]

	h1 = minheapify(x)
	#print(len(h1),h1)
	h2 = minheapify(a)
	#print(len(h2),h2)
	# Interestingly, when appending an item to, says, h1, 
	# it appends to ALL h1, results a growing no. of element during merge
	#hh = minmerge(0,h1,999999,h2,h1,h2,h1,h2,h1,h2,h1,h2,h1,h2,h1,h2)
	
	#hh = minmerge((0,'p'),h1,(999999,'king of the world'),h2)
	hh = minmerge(0,h1,999999,h2)
	# for _ in range(10):
		# print(minpop(hh))
		# print(hh)
	#print(len(hh),hh,'\n')
	print(h1)
	print([i for i in flatten0(h1)])
	
	import timeit
	#print(timeit.repeat(stmt="minheapify(a)", setup="from __main__ import a,minheapify", number=1000, repeat=3))
	# print(timeit.repeat(stmt="minmerge(0,h1,999999,h2)", setup="from __main__ import minmerge,h1,h2", number=1000, repeat=3))
	# print(timeit.repeat(stmt="minmerge1(0,h1,999999,h2)", setup="from __main__ import minmerge1,h1,h2", number=1000, repeat=3))
	# print(timeit.repeat(stmt="minpop(hh)", setup="from __main__ import hh,minpop", number=1000, repeat=3))
	#print(timeit.repeat(stmt="flatten(hh)", setup="from __main__ import hh,flatten3,flatten2,flatten", number=1, repeat=3))
	#print(timeit.repeat(stmt="x=[i for i in flatten3(hh)]", setup="from __main__ import hh,flatten3,flatten2,flatten", number=1, repeat=3))
	print(timeit.repeat(stmt="size(hh)", setup="from __main__ import hh,size", number=1, repeat=3))
	print(timeit.repeat(stmt="depth(hh)", setup="from __main__ import hh,depth", number=1, repeat=3))
	print(timeit.repeat(stmt="isMinpheap(hh)", setup="from __main__ import hh,isMinpheap", number=1, repeat=3))

	
	
