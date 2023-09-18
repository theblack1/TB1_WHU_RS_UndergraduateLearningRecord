
package ch04.usefulclass;

public class Person102 {

	String name;
	int age;

	public Person102(String name, int age) {
		this.name = name;
		this.age = age;
	}

	@Override
	public String toString() {
		return "Person [name=" + name + ", age=" + age + "]";
	}

	@Override
	public boolean equals(Object otherObject) {

		// 判断比较的参数也是Person类型
		if (otherObject instanceof Person102) {
			Person102 otherPerson = (Person102) otherObject;
			// 年龄作为比较规则
			if (this.age == otherPerson.age) {
				return true;
			}
		}
		return false;
	}

}
