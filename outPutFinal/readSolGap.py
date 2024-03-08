fi = open("allDoc.txt")
last='a'
for nm in fi:
	aux = nm.strip('\n\r') 
	f2 = open(aux)
	last_line = f2.readlines()[-1]
	value = last_line.split()
	pos = aux.find("_")
	temp = aux[0:pos+1]
	aux = aux.replace(temp,'')
	aux = aux.replace('.txt','')
	if(aux[0]!=last):
		print("")
	value[3] = value[3].replace('(','')
	value[3] = value[3].replace(')','')
	time = float(value[1])/1000
	print (aux, time,value[3],value[-1])
	last=aux[0]
	f2.close()
fi.close()
