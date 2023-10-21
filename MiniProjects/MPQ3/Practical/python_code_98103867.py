student_number = 98103867
Name = 'Mohammad'
Last_Name = 'Abolnejadian'







def get_sample_x_z1_z2():
    i = -3
    picked_z1 = -3
    sum = 0
    last_sum = 0
    random_number = random.random()
    for _ in range(25):
        last_sum = sum
        sum += get_p_z1(i)
        if  random_number >= last_sum and random_number <= sum:
            picked_z1 = i
            break
        i += 0.25

    i = -3
    picked_z2 = -3
    sum = 0
    last_sum = 0
    random_number = random.random()
    for _ in range(25):
        last_sum = sum
        sum += get_p_z2(i)
        if  random_number >= last_sum and random_number <= sum:
            picked_z2 = i
            break
        i += 0.25

    all_xi_probs = get_p_x_cond_z1_z2(picked_z1, picked_z2)
    image = []
    for x in range(NUM_PIXELS):
        if x % 28 == 0:
            image.append([])
        prob = all_xi_probs[x]
        random_number = random.random()
        if random_number <= prob:
            image[x//28].append(1)
        else:
            image[x//28].append(0)

    return image, picked_z1, picked_z2

for _ in range(5):
    image, z1, z2 = get_sample_x_z1_z2()
    plt.title(f'Z1:${z1}, Z2:${z2}')
    plt.imshow(image)
    plt.show()

def probs_to_image(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

i = -3
j = -3
all_zs = []
for z1_counter in range(25):
    all_zs.append([])
    j = -3
    for z2_counter in range(25):
        all_xi_probs = get_p_x_cond_z1_z2(i, j)
        all_zs[z1_counter].append(probs_to_image(all_xi_probs, 28))
        j += 0.25
    i += 0.25

fig, axs = plt.subplots(25, 25,figsize=(15,15))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Z2: -3 to 3")
plt.ylabel("Z1: -3 to 3")

for x in range(25):
    for y in range(25):
        axs[x, y].imshow(all_zs[x][y])
        axs[x, y].axis('off')


plt.show()




all_p_x_z1_z2 = []
for i in range(len(disc_z1)):
    all_p_x_z1_z2.append([])
    for j in range(len(disc_z2)):
        all_p_x_z1_z2[i].append(get_p_x_cond_z1_z2(disc_z1[i], disc_z2[j]))

def get_marginal_log_likelyhood(image):
    inverse_val_data = 1 - np.array(image)
    p = 0
    for i in range(len(disc_z1)):
        for j in range(len(disc_z2)):
            p += np.prod(np.absolute(inverse_val_data - all_p_x_z1_z2[i][j])) * get_p_z1(disc_z1[i]) * get_p_z2(disc_z2[j])
    if p == 0:
        return float('inf')
    return np.log10(p)

val_data_log_likelihoods = []
for val in val_data:
    val_likelyhood = get_marginal_log_likelyhood(val)
    val_data_log_likelihoods.append(val_likelyhood)
val_data_avg = np.average(val_data_log_likelihoods)
val_data_std = np.std(val_data_log_likelihoods)
print(f'validation data avaerage: {val_data_avg}')
print(f'validation data std: {val_data_std}')

def distance_test_from_val_avg(test_likelyhood):
    return np.absolute(test_likelyhood - val_data_avg)
real_images_likelyhoods = []
corrupted_images_likelyhoods = []
for test in test_data:
    test_likelyhood = get_marginal_log_likelyhood(test)
    if test_likelyhood == float('inf'):
        continue
    if distance_test_from_val_avg(test_likelyhood) > 3 * val_data_std:
        corrupted_images_likelyhoods.append(test_likelyhood)
    else:
        real_images_likelyhoods.append(test_likelyhood)

data = []
data.append(real_images_likelyhoods)
data.append(corrupted_images_likelyhoods)
titles = ["Real images marginal log likelyhoods", "Corrupted images marginal log likelyhoods"]


f,a = plt.subplots(2,1, figsize=(8,6))
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(data[idx], bins = 82)
    ax.set_title(titles[idx])
plt.tight_layout()

plt.show()



import pandas as pd
import random
import matplotlib.pyplot as plt





class BN(object):

    def __init__(self) -> None:

        self.n = 6 # We don't take G in counter, because it doesn't have any usage and any CPT.
        self.nodes_list = ['a', 'b', 'e', 'c', 'd', 'f']
        self.node_parents = {
            'a' : [],
            'b' : [],
            'c' : ['a', 'e'],
            'd' : ['a', 'c'],
            'e' : ['b'],
            'f' : ['d']
        }
        
        self.a_cpt = pd.DataFrame({
        'a' : [1, 0],
        'p': [0.8 , 0.2]})
        self.b_cpt = pd.DataFrame({
        'b' : [1, 0], 
        'p' : [0.55 , 0.45]})
        self.c_cpt = pd.DataFrame({
        'c' : [0, 1, 0, 1, 0, 1, 0, 1],
        'a' : [0, 0, 1, 1, 0, 0, 1, 1],
        'e' : [0, 0, 0, 0, 1, 1, 1, 1],
        'p' : [0.3, 0.7, 0.5, 0.5, 0.85, 0.15, 0.95, 0.05]})
        self.d_cpt = pd.DataFrame({
        'd' : [0, 1, 0, 1, 0, 1, 0, 1],
        'a' : [0, 0, 1, 1, 0, 0, 1, 1],
        'c' : [0, 0, 0, 0, 1, 1, 1, 1],
        'p' : [0.2, 0.8, 0.5, 0.5, 0.35, 0.65, 0.33, 0.67]
        })
        self.e_cpt = pd.DataFrame({
            'e' : [1, 1, 0, 0],
            'b' : [1, 0, 1, 0],
            'p' : [0.3, 0.9, 0.7, 0.1]
        })
        self.f_cpt = pd.DataFrame({
            'f' : [1, 1, 0, 0],
            'd' : [1, 0, 1, 0],
            'p' : [0.2, 0.25, 0.8, 0.75]
        })

        self.cpts = {
            'a' : self.a_cpt,
            'b' : self.b_cpt,
            'c' : self.c_cpt,
            'd' : self.d_cpt,
            'e' : self.e_cpt,
            'f' : self.f_cpt
        }
        cols = self.a_cpt.columns[:-1].intersection(self.c_cpt.columns[:-1]).tolist()
        ac = self.a_cpt.merge(self.c_cpt, on=cols)
        ac = ac.assign(p = ac.p_x*ac.p_y).drop(columns=['p_x', 'p_y'])

        cols = ac.columns[:-1].intersection(self.d_cpt.columns[:-1]).tolist()
        acd = ac.merge(self.d_cpt, on=cols)
        acd = acd.assign(p = acd.p_x*acd.p_y).drop(columns=['p_x', 'p_y'])

        cols = acd.columns[:-1].intersection(self.e_cpt.columns[:-1]).tolist()
        acde = acd.merge(self.e_cpt, on=cols)
        acde = acde.assign(p = acde.p_x*acde.p_y).drop(columns=['p_x', 'p_y'])

        cols = acde.columns[:-1].intersection(self.b_cpt.columns[:-1]).tolist()
        acdeb = acde.merge(self.b_cpt, on=cols)
        acdeb = acdeb.assign(p = acdeb.p_x*acdeb.p_y).drop(columns=['p_x', 'p_y'])

        cols = acdeb.columns[:-1].intersection(self.f_cpt.columns[:-1]).tolist()
        self.joint = acdeb.merge(self.f_cpt, on=cols)
        self.joint = self.joint.assign(p = self.joint.p_x*self.joint.p_y).drop(columns=['p_x', 'p_y'])
        
    
    def cpt(self, node, value):
        return self.cpts[node]
    
    def pmf(self, query, evidence) -> float:
        like_p = 0
        for _, row in self.joint.iterrows():
            for q in query:
                variable = q[0]
                value = q[1]
                if row[variable] != value:
                    break
            else:
                for e in evidence:
                    variable = e[0]
                    value = e[1]
                    if row[variable] != value:
                        break
                else:
                    like_p += row['p']

        normalizing_p = 0
        for _, row in self.joint.iterrows():
            for e in evidence:
                variable = e[0]
                value = e[1]
                if row[variable] != value:
                    break
            else:
                normalizing_p += row['p']

        return (like_p / normalizing_p)
        
    
    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin = 1e2) -> float:
        """
        Parameters
        ----------
        query: list
            list of variables an their values
            e.g. [('a', 0), ('e', 1)]
        evidence: list
            list of observed variables and their values
            e.g. [('b', 0), ('c', 1)]
        sampling_method:
            "Prior", "Rejection", "Likelihood Weighting", "Gibbs"
        num_iter:
            number of the generated samples 
        num_burnin:
            (used only in gibbs sampling) number of samples that we ignore at the start for gibbs method to converge
            
        Returns
        -------
        probability: float
            approximate P(query|evidence) calculated by sampling
        """
        
        if sampling_method.lower() == 'prior':
            prior_samples = []
            for _ in range(num_iter):
                sample = {}
                for node in self.nodes_list: 
                    cpt = self.cpt(node, _)
                    father_evidences = {}
                    for father in self.node_parents[node]:
                        father_evidences[father] = sample[father]
                    for _, row in cpt.iterrows():
                        for father_const in father_evidences:
                            if row[father_const] != father_evidences[father_const]:
                                break
                        else:
                            p = row['p']
                            rand = random.random()
                            if rand <= p:
                                sample[node] = row[node]
                            else:
                                sample[node] = 1 - row[node]
                            break
                prior_samples.append(sample)
            
            n = 0
            favourable = 0
            for sample in prior_samples:
                for e in evidence:
                    variable = e[0]
                    value = e[1]
                    if sample[variable] != value:
                        break
                else:
                    n += 1 #if matched evidence, then add one to total accepted samples
                    for q in query:
                        variable = q[0]
                        value = q[1]
                        if sample[variable] != value:
                            break
                    else:
                        favourable += 1 #if matched evidence and query, then add to favourable
            return favourable / n #return p
        
        elif sampling_method.lower() == 'rejection':
            evidence_vars = {}
            for e in evidence:
                evidence_vars[e[0]] = e[1]
            rejection_samples = []
            sample_iteration = 0
            while sample_iteration < num_iter:
                sample = {}
                reject_sample = False
                for node in self.nodes_list: 
                    if reject_sample:
                        break
                    cpt = self.cpt(node, "")
                    father_evidences = {}
                    for father in self.node_parents[node]:
                        father_evidences[father] = sample[father]
                    for _, row in cpt.iterrows():
                        for father_const in father_evidences:
                            if row[father_const] != father_evidences[father_const]:
                                break
                        else:
                            p = row['p']
                            rand = random.random()
                            if rand <= p:
                                sample[node] = row[node]
                            else:
                                sample[node] = 1 - row[node]
                            
                            if node in evidence_vars:
                                if evidence_vars[node] != sample[node]:
                                   reject_sample = True 
                            break
                if not reject_sample:
                    rejection_samples.append(sample)
                    sample_iteration += 1

            n = num_iter # because we know all the samples are consistant
            favourable = 0
            for sample in rejection_samples:
                for q in query:
                    variable = q[0]
                    value = q[1]
                    if sample[variable] != value:
                        break
                else:
                    favourable += 1 #if matched evidence and query, then add to favourable
            return favourable / n #return p
        
        elif sampling_method.lower() == 'likelihood weighting':
            evidence_vars = {}
            for e in evidence:
                evidence_vars[e[0]] = e[1]
            likelyhood_samples = []
            for _ in range(num_iter):
                sample = {}
                sample['weight'] = 1
                for node in self.nodes_list:
                    cpt = self.cpt(node, _)
                    father_evidences = {}
                    for father in self.node_parents[node]:
                        father_evidences[father] = sample[father]
                    if node in evidence_vars:
                        sample[node] = evidence_vars[node]
                        for _, row in cpt.iterrows():
                            if row[node] != evidence_vars[node]:
                                continue
                            for father_const in father_evidences:
                                if father_evidences[father_const] != row[father_const]:
                                    break
                            else:
                                w = row['p']
                                sample['weight'] *= w
                                break
                    else:
                        for _, row in cpt.iterrows():
                            for father_const in father_evidences:
                                if row[father_const] != father_evidences[father_const]:
                                    break
                            else:
                                p = row['p']
                                rand = random.random()
                                if rand <= p:
                                    sample[node] = row[node]
                                else:
                                    sample[node] = 1 - row[node]
                                break
                likelyhood_samples.append(sample)

            total_w = 0 # we should get the weighted average
            favourable = 0
            for sample in likelyhood_samples:
                total_w += sample['weight']
                for q in query:
                    variable = q[0]
                    value = q[1]
                    if sample[variable] != value:
                        break
                else:
                    favourable += sample['weight'] #if matched evidence and query, then add w to favourable
            return favourable / total_w #return p

        elif sampling_method.lower() == 'gibbs':

            evidence_vars = {}
            for e in evidence:
                evidence_vars[e[0]] = e[1]
            chosen_sample = {}
            for node in self.nodes_list:
                if node in evidence_vars:
                    chosen_sample[node] = evidence_vars[node]
                else:
                    rand = random.random()
                    if rand > 0.5:
                        chosen_sample[node] = 0
                    else:
                        chosen_sample[node] = 1
            
            number_of_samples = int(num_iter + num_burnin) #because we ignore num_burnin first of samples
            gibbs_samples = []

            for _ in range(number_of_samples):
                for node in self.nodes_list: #Sample for each node
                    if node in evidence_vars: #if evidence don't touch
                        continue
                    query_to_give_pmf = [(node, 0)]
                    evidence_to_give_pmf = []
                    for evi in chosen_sample:
                        if evi == node:
                            continue
                        evidence_to_give_pmf.append((evi, chosen_sample[evi]))
                    p = self.pmf(query_to_give_pmf, evidence_to_give_pmf)
                    rand = random.random()
                    if rand <= p:
                        chosen_sample[node] = 0
                    else:
                        chosen_sample[node] = 1
                sample = chosen_sample.copy()
                gibbs_samples.append(sample)
            
            
            throw_away_samples = int(num_burnin)
            gibbs_samples = gibbs_samples[throw_away_samples:]
            n = len(gibbs_samples)
            favourable = 0
            for sample in gibbs_samples:
                for q in query:
                    variable = q[0]
                    value = q[1]
                    if sample[variable] != value:
                        break
                else:
                    favourable += 1 #if matched evidence and query, then add to favourable
            return favourable / n #return p
            




bn = BN()
pmf_one = bn.pmf([('f', 1)], [('a', 1), ('e', 0)])
pmf_two = bn.pmf([('c', 0), ('b', 1)], [('f', 1), ('d', 0)])

sample_iters = [100, 500, 1000, 3000, 10_000]
sample_methods = ['prior', 'rejection', 'likelihood weighting', 'gibbs']
sample_results_q1 = {}
for sample_method in sample_methods:
    sample_results_q1[sample_method] = {}
    for iter in sample_iters:
        p = bn.sampling([('f', 1)], [('a', 1), ('e', 0)], sample_method, iter)
        sample_results_q1[sample_method][iter] = p

sample_results_q2 = {}
for sample_method in sample_methods:
    sample_results_q2[sample_method] = {}
    for iter in sample_iters:
        p = bn.sampling([('c', 0), ('b', 1)], [('f', 1), ('d', 0)], sample_method, iter)
        sample_results_q2[sample_method][iter] = p

print(f'P(F=1|A=1,E=0) = {pmf_one}')
print(f'P(C=0,B=1|F=1,D=0) = {pmf_two}')


print ('SAMPLING RESUTLS')
print('-> Query: P(F=1|A=1,E=0)')
print()
for result in sample_results_q1:
    print(f'Method: {result}')
    for iteration in sample_results_q1[result]:
        print(f'iter: {iteration}')
        print(f'P: {sample_results_q1[result][iteration]}')
    print()

print('-> Query: P(C=0,B=1|F=1,D=0)')
print()
for result in sample_results_q2:
    print(f'Method: {result}')
    for iteration in sample_results_q2[result]:
        print(f'iter: {iteration}')
        print(f'P: {sample_results_q2[result][iteration]}')
    print()


errors_q1 = []
for result in sample_results_q1:
    for iteration in sample_results_q1[result]:
        errors_q1.append(abs(pmf_one - sample_results_q1[result][iteration]))
errors_q2 = []
for result in sample_results_q2:
    for iteration in sample_results_q2[result]:
        errors_q2.append(abs(pmf_two - sample_results_q2[result][iteration]))

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20,12))

ax1.plot(sample_iters, errors_q1[0:5])
ax1.plot(sample_iters, errors_q1[5:10])
ax1.plot(sample_iters, errors_q1[10:15])
ax1.plot(sample_iters, errors_q1[15:20])
ax1.legend(["Prior Sampling", "Rejection Sampling", 'Likelihood Wighting', 'Gibbs Sampling'])
ax1.set_title("Errors for query: P(F=1|A=1,E=0)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("error")

ax2.plot(sample_iters, errors_q2[0:5])
ax2.plot(sample_iters, errors_q2[5:10])
ax2.plot(sample_iters, errors_q2[10:15])
ax2.plot(sample_iters, errors_q2[15:20])
ax2.legend(["Prior Sampling", "Rejection Sampling", 'Likelihood Wighting', 'Gibbs Sampling'])
ax2.set_title("Errors for query: P(C=0,B=1|F=1,D=0)")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("error")

plt.show()


num_burnins = [100, 200, 500, 1000, 2000]
num_iters_errors_q1 = []
for num_burnin in num_burnins:
    p = bn.sampling([('f', 1)], [('a', 1), ('e', 0)], 'gibbs', 500, num_burnin)
    num_iters_errors_q1.append(abs(p - pmf_one))

plt.plot(num_burnins, num_iters_errors_q1)
plt.title('Errors for query: P(F=1|A=1,E=0) [500 Iterations]')
plt.xlabel('Burnin value')
plt.ylabel('error')
plt.show()

num_iters_errors_q2 = []
for num_burnin in num_burnins:
    p = bn.sampling([('c', 0), ('b', 1)], [('f', 1), ('d', 0)], 'gibbs', 500, num_burnin)
    num_iters_errors_q2.append(abs(p - pmf_two))

plt.plot(num_burnins, num_iters_errors_q2)
plt.title('Errors for query: P(C=0,B=1|F=1,D=0) [500 Iterations]')
plt.xlabel('Burnin value')
plt.ylabel('error')
plt.show()

print("ERRORS")
print('-> Query: P(F=1|A=1,E=0)')
print()
for i in range(len(num_burnins)):
    print(f'Burnin value: {num_burnins[i]}')
    print(f'Error: {num_iters_errors_q1[i]}')
    print()
print('-> Query: P(C=0,B=1|F=1,D=0)')
print()
for i in range(len(num_burnins)):
    print(f'Burnin value: {num_burnins[i]}')
    print(f'Error: {num_iters_errors_q2[i]}')
    print()





def get_mean_towers_coor(time_step: int, tower_records: list):
    x = np.average([tower_coor[0] for tower_coor in tower_records[time_step]])
    y = np.average([tower_coor[1] for tower_coor in tower_records[time_step]])
    return x, y


def P_coor0(coor0):
    x0, y0 = coor0
    return scipy.stats.multivariate_normal.pdf([x0, y0], 
                            mean=moving_model.get('Peurto_coordinates'), cov=moving_model.get('INIT_COV'))



def P_coor_given_prevCoor(coor, prev_coor):

    p_x = expon.pdf(abs(prev_coor[0] - coor[0]), loc=0, scale=moving_model['X_STEP']) / 2 #half the time west and half east
    p_y = expon.pdf(abs(prev_coor[1] - coor[1]), loc=0, scale=moving_model['Y_STEP']) 

    return (p_x, p_y)
    
def P_towerCoor_given_coor(tower_coor, tower_std, coor):
    p_x = norm.pdf(tower_coor[0], coor[0], tower_std)
    p_y = norm.pdf(tower_coor[1], coor[1], tower_std)

    return (p_x, p_y)
    
    
def P_record_given_coor(rec, coor, towers_info):
    p_x, p_y = 1, 1
    index = 1
    for record in rec:
        std = towers_info[str(i)]['std']
        tower_p_x, tower_p_y = P_towerCoor_given_coor(record, std, coor)
        p_x *= tower_p_x
        p_y *= tower_p_y
        index += 1
    
    return(p_x, p_y)
        










max_Px, max_Py = 0, 0
interval, step = 20, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None

towers_mean_x1, towers_mean_y1 = get_mean_towers_coor(1, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):
        
         for x1 in range(int(towers_mean_x1 - interval), int(towers_mean_x1 + interval), step):
            for y1 in range(int(towers_mean_y1 - interval), int(towers_mean_y1 + interval), step):
                    
                coor0 = (x0, y0)
                coor1 = (x1, y1)

                rec0 = tower_records[0]
                rec1 = tower_records[1]

                p_coor_0 = P_coor0(coor0)

                p_coor_1_given_coor0_x, p_coor_1_given_coor0_y = P_coor_given_prevCoor(coor1, coor0)

                p_rec_0_given_coor_0_x,  p_rec_0_given_coor_0_y= P_record_given_coor(rec0, coor0, towers_info)

                p_rec_1_given_coor_1_x, p_rec_1_given_coor_1_y = P_record_given_coor(rec1, coor1, towers_info)

                p_x = p_coor_0 * p_coor_1_given_coor0_x * p_rec_0_given_coor_0_x * p_rec_1_given_coor_1_x
                p_y = p_coor_0 * p_coor_1_given_coor0_y * p_rec_0_given_coor_0_y * p_rec_1_given_coor_1_y

                if p_x > max_Px:
                    best_x0 = x0
                    best_x1 = x1
                    max_Px = p_x
                
                if p_y > max_Py:
                    best_y0 = y0
                    best_y1 = y1
                    max_Py = p_y
                    
            
coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))






max_Px, max_Py = 0, 0
interval, step = 15, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None
best_x2, best_y2 = None, None

towers_mean_x2, towers_mean_y2 = get_mean_towers_coor(2, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):
        
        for x1 in range(int(coor1_estimations[-1][0] - interval), int(coor1_estimations[-1][0] + interval), step):
            for y1 in range(int(coor1_estimations[-1][1] - interval), int(coor1_estimations[-1][1] + interval), step):

                for x2 in range(int(towers_mean_x2 - interval), int(towers_mean_x2 + interval), step):
                    for y2 in range(int(towers_mean_y2 - interval), int(towers_mean_y2 + interval), step):

                        coor0 = (x0, y0)
                        coor1 = (x1, y1)
                        coor2 = (x2, y2)

                        rec0 = tower_records[0]
                        rec1 = tower_records[1]
                        rec2 = tower_records[2]

                        p_coor_0 = P_coor0(coor0)

                        p_rec_0_given_coor_0_x,  p_rec_0_given_coor_0_y= P_record_given_coor(rec0, coor0, towers_info)

                        p_rec_1_given_coor_1_x, p_rec_1_given_coor_1_y = P_record_given_coor(rec1, coor1, towers_info)

                        p_rec_2_given_coor_2_x, p_rec_2_given_coor_2_y = P_record_given_coor(rec2, coor2, towers_info)

                        p_coor_1_given_coor0_x, p_coor_1_given_coor0_y = P_coor_given_prevCoor(coor1, coor0)

                        p_coor_2_given_coor1_x, p_coor_2_given_coor1_y = P_coor_given_prevCoor(coor2, coor1)

                        p_x = p_coor_0 * p_rec_0_given_coor_0_x * p_rec_1_given_coor_1_x * p_rec_2_given_coor_2_x * p_coor_1_given_coor0_x * p_coor_2_given_coor1_x
                        p_y = p_coor_0 * p_rec_0_given_coor_0_y * p_rec_1_given_coor_1_y * p_rec_2_given_coor_2_y * p_coor_1_given_coor0_y * p_coor_2_given_coor1_y

                        if p_x > max_Px:
                            best_x0 = x0
                            best_x1 = x1
                            best_x2 = x2
                            max_Px = p_x

                        if p_y > max_Py:
                            best_y0 = y0
                            best_y1 = y1
                            best_y2 = y2
                            max_Py = p_y 


coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))
coor2_estimations.append((best_x2, best_y2))            



print(f'real_coor0: {real_coor(0)} - Estimated_coor0: {best_x0, best_y0}')
print(f'Estimation_error: {dist((best_x0, best_y0), real_coor(0))}')
print()
print(f'real_coor1: {real_coor(1)} - Estimated_coor1: {best_x1, best_y1}')
print(f'Estimation_error: {dist((best_x1, best_y1), real_coor(1))}')
print()
print(f'real_coor2: {real_coor(2)} - Estimated_coor2: {best_x2, best_y2}')
print(f'Estimation_error: {dist((best_x2, best_y2), real_coor(2))}')


max_Px, max_Py = 0, 0
interval, step = 10, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None
best_x2, best_y2 = None, None
best_x3, best_y3 = None, None

towers_mean_x3, towers_mean_y3 = get_mean_towers_coor(3, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):

        for x1 in range(int(coor1_estimations[-1][0] - interval), int(coor1_estimations[-1][0] + interval), step):
            for y1 in range(int(coor1_estimations[-1][1] - interval), int(coor1_estimations[-1][1] + interval), step):

                for x2 in range(int(coor2_estimations[-1][0] - interval), int(coor2_estimations[-1][0] + interval), step):
                    for y2 in range(int(coor2_estimations[-1][1] - interval), int(coor2_estimations[-1][1] + interval), step):

                        for x3 in range(int(towers_mean_x3 - interval), int(towers_mean_x3 + interval), step):
                            for y3 in range(int(towers_mean_y3 - interval), int(towers_mean_y3 + interval), step):

                                coor0 = (x0, y0)
                                coor1 = (x1, y1)
                                coor2 = (x2, y2)
                                coor3 = (x3, y3)

                                rec0 = tower_records[0]
                                rec1 = tower_records[1]
                                rec2 = tower_records[2]
                                rec3 = tower_records[3]

                                p_coor_0 = P_coor0(coor0)

                                p_rec_0_given_coor_0_x,  p_rec_0_given_coor_0_y= P_record_given_coor(rec0, coor0, towers_info)

                                p_rec_1_given_coor_1_x, p_rec_1_given_coor_1_y = P_record_given_coor(rec1, coor1, towers_info)

                                p_rec_2_given_coor_2_x, p_rec_2_given_coor_2_y = P_record_given_coor(rec2, coor2, towers_info)

                                p_rec_3_given_coor_3_x, p_rec_3_given_coor_3_y = P_record_given_coor(rec3, coor3, towers_info)

                                p_coor_1_given_coor0_x, p_coor_1_given_coor0_y = P_coor_given_prevCoor(coor1, coor0)

                                p_coor_2_given_coor1_x, p_coor_2_given_coor1_y = P_coor_given_prevCoor(coor2, coor1)

                                p_coor_3_given_coor2_x, p_coor_3_given_coor2_y = P_coor_given_prevCoor(coor3, coor2)

                                p_x = (p_coor_0 * p_rec_0_given_coor_0_x * p_rec_1_given_coor_1_x * p_rec_2_given_coor_2_x
                                * p_rec_3_given_coor_3_x * p_coor_1_given_coor0_x * p_coor_2_given_coor1_x * p_coor_3_given_coor2_x)

                                p_y = (p_coor_0 * p_rec_0_given_coor_0_y * p_rec_1_given_coor_1_y * p_rec_2_given_coor_2_y
                                * p_rec_3_given_coor_3_y * p_coor_1_given_coor0_y * p_coor_2_given_coor1_y * p_coor_3_given_coor2_y)

                                if p_x > max_Px:
                                    best_x0 = x0
                                    best_x1 = x1
                                    best_x2 = x2
                                    best_x3 = x3
                                    max_Px = p_x

                                if p_y > max_Py:
                                    best_y0 = y0
                                    best_y1 = y1
                                    best_y2 = y2
                                    best_y3 = y3
                                    max_Py = p_y  


coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))
coor2_estimations.append((best_x2, best_y2)) 
coor3_estimations.append((best_x3, best_y3)) 



print(f'real_coor0: {real_coor(0)} - Estimated_coor0: {best_x0, best_y0}')
print(f'Estimation_error: {dist((best_x0, best_y0), real_coor(0))}')
print()
print(f'real_coor1: {real_coor(1)} - Estimated_coor1: {best_x1, best_y1}')
print(f'Estimation_error: {dist((best_x1, best_y1), real_coor(1))}')
print()
print(f'real_coor2: {real_coor(2)} - Estimated_coor2: {best_x2, best_y2}')
print(f'Estimation_error: {dist((best_x2, best_y2), real_coor(2))}')
print()
print(f'real_coor3: {real_coor(3)} - Estimated_coor3: {best_x3, best_y3}')
print(f'Estimation_error: {dist((best_x3, best_y3), real_coor(3))}')

errors_coor_0 = []
for zero in coor0_estimations:
    errors_coor_0.append(dist(zero, real_coor(0)))

errors_coor_1 = []
for one in coor1_estimations:
    errors_coor_1.append(dist(one, real_coor(1)))

errors_coor_2 = []
for two in coor2_estimations:
    errors_coor_2.append(dist(two, real_coor(2)))

errors_coor_3 = []
for three in coor3_estimations:
    errors_coor_3.append(dist(three, real_coor(3)))

print(f'Errors coor_0 : {errors_coor_0}')
print(f'Errors coor_1 : {errors_coor_1}')
print(f'Errors coor_2 : {errors_coor_2}')
print(f'Errors coor_3 : {errors_coor_3}')

x_axis = [1, 2, 3, 4]
plt.plot(x_axis, errors_coor_0)
plt.plot(x_axis[1:], errors_coor_1)
plt.plot(x_axis[2:], errors_coor_2)
plt.scatter(x_axis[3:], errors_coor_3)
plt.title("Errors for each coordination as we move forward")
plt.xlabel("#records")
plt.ylabel("Error")
plt.legend(["coor_0", "coor_1", "coor_2", "coor_3"])
plt.show()



