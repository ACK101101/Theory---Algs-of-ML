def findDecision(obj): #obj[0]: CompPrice, obj[1]: Income, obj[2]: Advertising, obj[3]: Population, obj[4]: Price, obj[5]: ShelveLoc, obj[6]: Age, obj[7]: Education, obj[8]: Urban, obj[9]: US
	# {"feature": "ShelveLoc", "instances": 282, "metric_value": 0.387, "depth": 1}
	if obj[5] == 'Medium':
		# {"feature": "Advertising", "instances": 149, "metric_value": 0.3988, "depth": 2}
		if obj[2]<=6.1610738255033555:
			# {"feature": "Price", "instances": 87, "metric_value": 0.2896, "depth": 3}
			if obj[4]>95.23156457844308:
				# {"feature": "CompPrice", "instances": 74, "metric_value": 0.2215, "depth": 4}
				if obj[0]<=127.6891891891892:
					# {"feature": "Education", "instances": 38, "metric_value": 0.0468, "depth": 5}
					if obj[7]>11:
						return 'No'
					elif obj[7]<=11:
						# {"feature": "Income", "instances": 9, "metric_value": 0.1111, "depth": 6}
						if obj[1]<=57:
							return 'No'
						elif obj[1]>57:
							# {"feature": "Population", "instances": 2, "metric_value": 0.0, "depth": 7}
							if obj[3]<=45:
								return 'No'
							elif obj[3]>45:
								return 'Yes'
							else: return 'Yes'
						else: return 'No'
					else: return 'No'
				elif obj[0]>127.6891891891892:
					# {"feature": "Age", "instances": 36, "metric_value": 0.3093, "depth": 5}
					if obj[6]>37.990568555249524:
						# {"feature": "US", "instances": 29, "metric_value": 0.2541, "depth": 6}
						if obj[9] == 'No':
							# {"feature": "Population", "instances": 19, "metric_value": 0.3275, "depth": 7}
							if obj[3]>104:
								# {"feature": "Income", "instances": 18, "metric_value": 0.3175, "depth": 8}
								if obj[1]>31:
									# {"feature": "Education", "instances": 14, "metric_value": 0.3429, "depth": 9}
									if obj[7]>11:
										# {"feature": "Urban", "instances": 10, "metric_value": 0.475, "depth": 10}
										if obj[8] == 'Yes':
											return 'No'
										elif obj[8] == 'No':
											return 'Yes'
										else: return 'Yes'
									elif obj[7]<=11:
										return 'No'
									else: return 'No'
								elif obj[1]<=31:
									return 'No'
								else: return 'No'
							elif obj[3]<=104:
								return 'Yes'
							else: return 'Yes'
						elif obj[9] == 'Yes':
							return 'No'
						else: return 'No'
					elif obj[6]<=37.990568555249524:
						# {"feature": "Population", "instances": 7, "metric_value": 0.0, "depth": 6}
						if obj[3]>89:
							return 'Yes'
						elif obj[3]<=89:
							return 'No'
						else: return 'No'
					else: return 'Yes'
				else: return 'No'
			elif obj[4]<=95.23156457844308:
				# {"feature": "CompPrice", "instances": 13, "metric_value": 0.2308, "depth": 4}
				if obj[0]<=108:
					# {"feature": "Age", "instances": 8, "metric_value": 0.2143, "depth": 5}
					if obj[6]>34:
						# {"feature": "Income", "instances": 7, "metric_value": 0.1429, "depth": 6}
						if obj[1]<=93:
							return 'No'
						elif obj[1]>93:
							# {"feature": "Population", "instances": 2, "metric_value": 0.0, "depth": 7}
							if obj[3]>76:
								return 'No'
							elif obj[3]<=76:
								return 'Yes'
							else: return 'Yes'
						else: return 'No'
					elif obj[6]<=34:
						return 'Yes'
					else: return 'Yes'
				elif obj[0]>108:
					return 'Yes'
				else: return 'Yes'
			else: return 'Yes'
		elif obj[2]>6.1610738255033555:
			# {"feature": "Price", "instances": 62, "metric_value": 0.4221, "depth": 3}
			if obj[4]>115.96774193548387:
				# {"feature": "Education", "instances": 32, "metric_value": 0.3698, "depth": 4}
				if obj[7]<=15:
					# {"feature": "Age", "instances": 20, "metric_value": 0.375, "depth": 5}
					if obj[6]>36:
						# {"feature": "CompPrice", "instances": 16, "metric_value": 0.3409, "depth": 6}
						if obj[0]>123:
							# {"feature": "Population", "instances": 11, "metric_value": 0.3409, "depth": 7}
							if obj[3]>125:
								# {"feature": "US", "instances": 8, "metric_value": 0.3571, "depth": 8}
								if obj[9] == 'Yes':
									# {"feature": "Income", "instances": 7, "metric_value": 0.3429, "depth": 9}
									if obj[1]>60:
										# {"feature": "Urban", "instances": 5, "metric_value": 0.4, "depth": 10}
										if obj[8] == 'Yes':
											return 'No'
										elif obj[8] == 'No':
											return 'No'
										else: return 'No'
									elif obj[1]<=60:
										return 'No'
									else: return 'No'
								elif obj[9] == 'No':
									return 'Yes'
								else: return 'Yes'
							elif obj[3]<=125:
								return 'Yes'
							else: return 'Yes'
						elif obj[0]<=123:
							return 'No'
						else: return 'No'
					elif obj[6]<=36:
						return 'Yes'
					else: return 'Yes'
				elif obj[7]>15:
					# {"feature": "CompPrice", "instances": 12, "metric_value": 0.1333, "depth": 5}
					if obj[0]<=134:
						return 'No'
					elif obj[0]>134:
						# {"feature": "Income", "instances": 5, "metric_value": 0.2, "depth": 6}
						if obj[1]<=51:
							return 'No'
						elif obj[1]>51:
							# {"feature": "Population", "instances": 2, "metric_value": 0.0, "depth": 7}
							if obj[3]>171:
								return 'Yes'
							elif obj[3]<=171:
								return 'No'
							else: return 'No'
						else: return 'Yes'
					else: return 'No'
				else: return 'No'
			elif obj[4]<=115.96774193548387:
				# {"feature": "Age", "instances": 30, "metric_value": 0.2824, "depth": 4}
				if obj[6]>54.9:
					# {"feature": "CompPrice", "instances": 17, "metric_value": 0.362, "depth": 5}
					if obj[0]<=124:
						# {"feature": "Income", "instances": 13, "metric_value": 0.2577, "depth": 6}
						if obj[1]<=71:
							# {"feature": "Population", "instances": 8, "metric_value": 0.1667, "depth": 7}
							if obj[3]<=267:
								return 'No'
							elif obj[3]>267:
								# {"feature": "Education", "instances": 3, "metric_value": 0.3333, "depth": 8}
								if obj[7]<=12:
									# {"feature": "Urban", "instances": 2, "metric_value": 0.5, "depth": 9}
									if obj[8] == 'Yes':
										# {"feature": "US", "instances": 2, "metric_value": 0.5, "depth": 10}
										if obj[9] == 'Yes':
											return 'Yes'
										else: return 'Yes'
									else: return 'Yes'
								elif obj[7]>12:
									return 'No'
								else: return 'No'
							else: return 'No'
						elif obj[1]>71:
							# {"feature": "Population", "instances": 5, "metric_value": 0.0, "depth": 7}
							if obj[3]<=416:
								return 'Yes'
							elif obj[3]>416:
								return 'No'
							else: return 'No'
						else: return 'Yes'
					elif obj[0]>124:
						return 'Yes'
					else: return 'Yes'
				elif obj[6]<=54.9:
					return 'Yes'
				else: return 'Yes'
			else: return 'Yes'
		else: return 'Yes'
	elif obj[5] == 'Bad':
		# {"feature": "Price", "instances": 74, "metric_value": 0.1955, "depth": 2}
		if obj[4]>92.45040787018631:
			# {"feature": "Advertising", "instances": 60, "metric_value": 0.1086, "depth": 3}
			if obj[2]<=15:
				# {"feature": "Age", "instances": 54, "metric_value": 0.0679, "depth": 4}
				if obj[6]>50.53703703703704:
					return 'No'
				elif obj[6]<=50.53703703703704:
					# {"feature": "Income", "instances": 24, "metric_value": 0.1389, "depth": 5}
					if obj[1]>63:
						# {"feature": "Population", "instances": 12, "metric_value": 0.1667, "depth": 6}
						if obj[3]>276:
							return 'No'
						elif obj[3]<=276:
							# {"feature": "CompPrice", "instances": 4, "metric_value": 0.3333, "depth": 7}
							if obj[0]>121:
								# {"feature": "Education", "instances": 3, "metric_value": 0.3333, "depth": 8}
								if obj[7]>14:
									# {"feature": "Urban", "instances": 2, "metric_value": 0.5, "depth": 9}
									if obj[8] == 'Yes':
										# {"feature": "US", "instances": 2, "metric_value": 0.5, "depth": 10}
										if obj[9] == 'Yes':
											return 'Yes'
										else: return 'Yes'
									else: return 'Yes'
								elif obj[7]<=14:
									return 'No'
								else: return 'No'
							elif obj[0]<=121:
								return 'Yes'
							else: return 'Yes'
						else: return 'Yes'
					elif obj[1]<=63:
						return 'No'
					else: return 'No'
				else: return 'No'
			elif obj[2]>15:
				# {"feature": "Income", "instances": 6, "metric_value": 0.0, "depth": 4}
				if obj[1]<=65:
					return 'No'
				elif obj[1]>65:
					return 'Yes'
				else: return 'Yes'
			else: return 'No'
		elif obj[4]<=92.45040787018631:
			# {"feature": "Income", "instances": 14, "metric_value": 0.125, "depth": 3}
			if obj[1]<=81:
				# {"feature": "CompPrice", "instances": 8, "metric_value": 0.0, "depth": 4}
				if obj[0]<=123:
					return 'No'
				elif obj[0]>123:
					return 'Yes'
				else: return 'Yes'
			elif obj[1]>81:
				return 'Yes'
			else: return 'Yes'
		else: return 'Yes'
	elif obj[5] == 'Good':
		# {"feature": "Price", "instances": 59, "metric_value": 0.2811, "depth": 2}
		if obj[4]<=146.67252077715062:
			# {"feature": "US", "instances": 49, "metric_value": 0.1997, "depth": 3}
			if obj[9] == 'Yes':
				# {"feature": "Education", "instances": 37, "metric_value": 0.0954, "depth": 4}
				if obj[7]>14:
					return 'Yes'
				elif obj[7]<=14:
					# {"feature": "Age", "instances": 17, "metric_value": 0.1412, "depth": 5}
					if obj[6]<=61:
						return 'Yes'
					elif obj[6]>61:
						# {"feature": "Population", "instances": 5, "metric_value": 0.0, "depth": 6}
						if obj[3]<=260:
							return 'Yes'
						elif obj[3]>260:
							return 'No'
						else: return 'No'
					else: return 'Yes'
				else: return 'Yes'
			elif obj[9] == 'No':
				# {"feature": "Age", "instances": 12, "metric_value": 0.3333, "depth": 4}
				if obj[6]<=63:
					# {"feature": "Income", "instances": 9, "metric_value": 0.3333, "depth": 5}
					if obj[1]>39:
						# {"feature": "Population", "instances": 6, "metric_value": 0.1667, "depth": 6}
						if obj[3]<=264:
							return 'Yes'
						elif obj[3]>264:
							# {"feature": "CompPrice", "instances": 2, "metric_value": 0.0, "depth": 7}
							if obj[0]>111:
								return 'No'
							elif obj[0]<=111:
								return 'Yes'
							else: return 'Yes'
						else: return 'No'
					elif obj[1]<=39:
						# {"feature": "Population", "instances": 3, "metric_value": 0.0, "depth": 6}
						if obj[3]>14:
							return 'No'
						elif obj[3]<=14:
							return 'Yes'
						else: return 'Yes'
					else: return 'No'
				elif obj[6]>63:
					return 'No'
				else: return 'No'
			else: return 'No'
		elif obj[4]>146.67252077715062:
			# {"feature": "CompPrice", "instances": 10, "metric_value": 0.1778, "depth": 3}
			if obj[0]<=156:
				# {"feature": "Age", "instances": 9, "metric_value": 0.1111, "depth": 4}
				if obj[6]>29:
					return 'No'
				elif obj[6]<=29:
					# {"feature": "Income", "instances": 2, "metric_value": 0.0, "depth": 5}
					if obj[1]>21:
						return 'Yes'
					elif obj[1]<=21:
						return 'No'
					else: return 'No'
				else: return 'Yes'
			elif obj[0]>156:
				return 'Yes'
			else: return 'Yes'
		else: return 'No'
	else: return 'Yes'
