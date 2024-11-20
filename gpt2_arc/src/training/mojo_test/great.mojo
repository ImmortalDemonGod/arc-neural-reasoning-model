from sys.param_env import env_get_string

fn main():
    var name = env_get_string["name", "No Name Provided"]()
    print("Hello, " + name + "!")
