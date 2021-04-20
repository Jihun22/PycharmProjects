import tensorflow as tf

v1 = tf.Variable(1.)
v2 = tf.Variable(2.)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)

	#print(name)
	print("\n\nv1 name:", v1.name)
	print("v2 name:", v2.name)
	v1_1, v2_1 = sess.run(["Variable:0", "Variable_1:0"])
	print("v1 value: {}".format(v1_1))
	print("v2 value: {}\n\n".format(v2_1))


#-----------------------------------------------
# 2. naming: name을 직접 설정하는 방법
#-----------------------------------------------
v3 = tf.Variable(3.,name="v3")
v4 = tf.Variable(4.,name="v4")
v5 = tf.get_variable("v5",1,initializer=tf.constant_initializer(5.))
v6 = tf.get_variable("v6",1,initializer=tf.constant_initializer(6.))

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)

	print("\n\nv3 name: {}".format(v3.name))
	print("v4 name: {}".format(v4.name))
	print("v5 name: {}".format(v5.name))
	print("v6 name: {}\n\n".format(v6.name))

	v3_1, v4_1, v5_1, v6_1 = sess.run(["v3:0", "v4:0", "v5:0", "v6:0"])
	print("\n\nv3 value: {}".format(v3_1))
	print("v4 value: {}".format(v4_1))
	print("v5 value: {}".format(v5_1))
	print("v6 value: {}\n\n".format(v6_1))

#-----------------------------------------------
# 3. name_scope()
#-----------------------------------------------
v3 = tf.Variable(3., name="v3")
v4 = tf.Variable(4., name="v4")

print("\n\nv3 name: {}".format(v3.name))
print("v4 name: {}".format(v4.name))

with tf.name_scope("scope1"):
	v3 = tf.Variable(3., name="v3")
	v4 = tf.Variable(4., name="v4")
	v3_1 = tf.get_variable("v3",1,initializer=tf.constant_initializer(3.))
	v4_1 = tf.get_variable("v4",1,initializer=tf.constant_initializer(4.))

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)

	print("\n\nv3 name: {}".format(v3.name))
	print("v4 name: {}".format(v4.name))

	print("\n\nv3 value: {}".format(sess.run(v3_1)))
	print("v4 value: {}".format(sess.run(v4_1)))

#-----------------------------------------------
# 4. variable_scope()
#-----------------------------------------------
with tf.variable_scope("scope2"):
        v1 = tf.get_variable("v1",1,initializer=tf.constant_initializer(1.))
        v2 = tf.get_variable("v2",1,initializer=tf.constant_initializer(2.))
        v3 = tf.Variable(3.,name="v3")
        v4 = tf.Variable(4.,name="v4")

print("\n\nv1 name: {}".format(v1.name))
print("v2 name: {}".format(v2.name))
print("v3 name: {}".format(v3.name))
print("v4 name: {}".format(v4.name))

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	print("\n\nv1 name: {}".format(v1.name))
	print("v2 name: {}".format(v2.name))
	print("v3 name: {}".format(v3.name))
	print("v4 name: {}".format(v4.name))

	print("\n\nv1 name: {}".format(sess.run(v1.name)))
	print("v2 name: {}".format(sess.run(v2.name)))
	print("v3 name: {}".format(sess.run(v3.name)))
	print("v4 name: {}".format(sess.run(v4.name)))

	print("\n\nv1 : {}".format(sess.run(v1)))
	print("v2 : {}".format(sess.run(v2)))
	print("v3 : {}".format(sess.run(v3)))
	print("v4 : {}".format(sess.run(v4)))

#-----------------------------------------------
# 5. variable_sharing
#-----------------------------------------------
with tf.variable_scope("scope2", reuse=True):
	v1_ = tf.get_variable("v1")
	v2_ = tf.get_variable("v2")

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)

	print("\n\nv1 name by sharing: {}".format(v1_.name))
	print("v2 name by sharing: {}".format(v2_.name))

	v1_1, v2_1, v1_2, v2_2 = sess.run([scope+"/v1:0", scope+"/v2:0", v1_, v2_])

	print('\n\nv1 values by name: {}'.format(v1_1))
	print('v2 values by name: {}'.format(v2_1))
	print('v1 values by sharing: {}'.format(v1_2))
	print('v2 values by sharing: {}\n\n'.format(v2_2))
