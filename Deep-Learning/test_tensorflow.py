import tensorflow as tf

def constant():
    print("Constant:")
    # Create TensorFlow object called tensor
    hello_constant = tf.constant('Hello World!')
    
    with tf.Session() as sess:
        # Run the tf.constant operation in the session
        output = sess.run(hello_constant).decode()      # without decode this will return b'Hello World!'
        print(output)


def placeholder():
    print("\nPlaceholder:")
    output = None
    x = tf.placeholder(tf.int32)    # designate x as a placeholder for a 0-dimensional tensor of type int32
    y = tf.placeholder(tf.float32)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123, y: 2.5})    # feed the value 123 into the x tensor and 2.5 into y
        
    print(output)   # output is assigned to x
    
 
def basic_math():
    print("\nBasic math:")
    a = 5
    b = 2
    c = 8
    x = tf.add(a,b)
    y = tf.subtract(a,b)
    z = tf.multiply(a,b)
    w = tf.subtract(tf.divide(c,b), 1)
    
    with tf.Session() as sess:
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run(z))
        print(sess.run(w))
        
        
def type_conversion():
    print("\nType Conversion:")
    x = tf.constant(2.0)   # can't subtract these in TF because they are different types
    y = tf.constant(1)
    
    # z = tf.subtract(x,y)   <-- TypeError
    z = tf.subtract(tf.cast(x, tf.int32), y)    # Cast x to int32
    
    with tf.Session() as sess:
        print(sess.run(z))

if __name__ == "__main__":
    constant()
    placeholder()
    basic_math()
    type_conversion()