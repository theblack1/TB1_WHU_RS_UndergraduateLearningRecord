package department;

import java.util.Vector;

import main.Stage;
import person.Ruler;

public class Admin extends Stage {
	Vector<Ruler> employeeList = new Vector<Ruler>();

	public Admin() {
	}

	public Admin(String name2, Ruler boss2, Ruler firstemployee) {
		name = name2;

		boss2.level = Ruler.L22;
		boss = boss2;
		firstemployee.level = Ruler.L25;

		employeeList.add(firstemployee);

		ShowMe();
	}

	@Override
	public void ShowMe() {
		System.out.println();
		System.out.println("Adim:" + name + "\nAdministrator:" + boss.name);
		System.out.println("Employee:");
		for (Ruler s : employeeList) {
			System.out.println(s.name);
		}

	}

}
