# One tracked moving object, modelled as a disc with constant velocity.
# Position and velocity are expressed in the enclosing array's header frame;
# their z components are ignored.

uint32 id
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
float32 radius
